# Chess Elo Prediction — Development Notes

*A record of the decisions, mistakes, and improvements made across v1 through v4 of this project.*

---

## Where we started and what went wrong

The first real run loaded 1,000 games, applied a `TIME_CONTROL = '600+0'` filter, and ended up with **128 games** to train on. The model reported a test MAE of 1,630 Elo — essentially useless. To put that in context: if you just predicted the dataset mean (roughly 1,600) for every single game, you'd get almost exactly that MAE. The model had learned nothing.

The diagnosis was straightforward once we looked at the numbers. Two things compounded each other:

**The time-control filter was catastrophic.** The Lichess database contains games across all time controls — blitz, rapid, classical, bullet. Filtering strictly to `600+0` (10 minutes, no increment) kept roughly 10% of available games. With only 1,000 games loaded to begin with, that left 128 after all filtering. You cannot train a neural network on 128 examples.

**The model never actually trained.** The attention heatmap output showed `pred 0` — the model's output was literally zero. With a dataset this small, early stopping triggered after just a few epochs, the loss barely moved from its initial value, and the checkpoint saved a model that had converged to predicting the mean.

The fix for both problems was the same: remove the time-control filter (or at minimum only enforce a minimum base time), and load significantly more games. Setting `TIME_CONTROL = None` with `TC_BASE_MIN = 180` (keeping games with at least a 3-minute base) retained about 60-70% of the data, giving us well over 500k games at the 1M load setting.

---

## The data pipeline choices

### Why stream-parse rather than load the whole file

The Lichess monthly dumps are multi-gigabyte compressed files. Loading even a single month fully into memory isn't practical on a laptop. The PGN parser reads line-by-line through the decompressed stream and stops as soon as it hits `MAX_GAMES` complete games. This means loading 200k games from a file that contains 10M takes about 8 seconds rather than several minutes.

### Feature caching

Board feature extraction — the python-chess replay step — takes roughly 15 minutes per 100k games on a modern 8-core laptop. This only needs to happen once. Both the board features and clock features are saved as pickle files to `cache/`, keyed by dataset size. On every subsequent run they load in a few seconds. This is one of those changes that sounds minor but completely changes the development workflow — you can iterate on the model without waiting 15 minutes at the start of every run.

The AE embeddings are cached the same way, additionally keyed by `AE_LATENT_DIM` so changing the bottleneck size automatically triggers a retrain.

### The time-control normalisation for clock features

Clock times are normalised by dividing by the base time control. A player who spends 30 seconds on a move in a 10-minute game is behaving very differently from one who spends 30 seconds in a 3-minute game. Without this normalisation, the clock features would be dominated by the time control itself rather than player behaviour, making them useless for comparing across different game lengths.

The chess.com PGN format also caused a bug here: chess.com writes time control as `"600"` (no `+0`), while Lichess writes `"600+0"`. The original regex `r'(\d+)\+(\d+)'` silently failed to match the chess.com format and defaulted `tc_base` to 300, meaning all clock features were normalised against the wrong value. The fix was to add a fallback for the no-increment format.

---

## The vocabulary

Each unique chess move (e.g. `e4`, `Nf3`, `Bxc6+`, `O-O`) gets an integer ID. At 1M games this vocabulary grows to about 10,000-11,000 unique tokens — there are a lot of move variations in algebraic notation when you include check symbols, captures, disambiguations, and promotions.

Two special tokens: `PAD=0` for padding sequences to equal length within a batch, and `UNK=1` for moves not seen during training (rare, but possible if you run inference on unusual games).

In v4 this was simplified from a class to a plain dictionary with standalone functions (`add_move`, `encode_move`, `vocab_size`). The class wasn't adding anything useful — it was just wrapping a dict in a container that required understanding the class interface to use. The flat functions are easier to read and do exactly the same thing.

---

## Why distribution targets instead of regression

This was directly inspired by the Ouzounis repo, which won second place in an AI competition with this approach. The core insight is about the nature of chess ratings.

A player rated 1600 does not always play like a 1600. In any given game, their performance varies — sometimes they're sharp and play 1800-level chess, sometimes they miss things a 1400 wouldn't. This game-to-game variance is not noise to be minimised; it's a fundamental property of the rating system.

The Elo formula itself implies a standard deviation. If a player rated 100 points higher wins 64% of the time, and you work backwards through the normal distribution implied by that, you get σ ≈ 200 Elo points. The Ouzounis Decisions_Explained.md has the full derivation — it's a clean piece of mathematics worth reading.

So instead of training the model to output a single number and measuring loss against the true Elo, we:
1. Convert each player's true Elo into a Gaussian distribution over 40 rating bins (0–4000, 100 points each) with σ=200
2. Train the model to output a probability distribution over those same 40 bins
3. Use KL-divergence as the loss function — it measures how different two distributions are
4. Recover the point estimate as the weighted mean of the predicted distribution

The practical advantages: you get uncertainty for free (wide distribution = uncertain, narrow = confident), the training signal is smoother (a prediction 200 points off contributes loss from several bins rather than just one), and the visualisation is genuinely more informative than a single number.

---

## The sequence autoencoder

The BiLSTM branch adds meaningful signal — the ablation analysis consistently shows sequence information contributing around 10-15% of feature importance compared to the static features. But on CPU, running the LSTM on every batch of every epoch is slow. For 200k games at `BATCH_SIZE=64`, that's 3,000 LSTM forward passes per epoch.

The autoencoder is a practical solution to this. Train it once (2 epochs on 15% of the data, about 5 minutes), cache the 32-d embeddings for every game, then use those cached vectors as extra static features in the main model. The main training loop then runs at the same speed as the static-only mode — no recurrence at runtime.

The architecture is a standard LSTM encoder-decoder:
- **Encoder**: BiLSTM → mean-pool over non-padding positions → linear projection → tanh → 32-d vector
- **Decoder**: project latent vector to LSTM hidden state → generate tokens with teacher forcing (feed ground-truth tokens at each step during training)

The training objective is token-level cross-entropy — predict the next move in the sequence from the compressed representation. This forces the encoder to learn which aspects of a move sequence are actually informative about playing style.

**A note on the AE loss numbers.** Epoch 1 loss around 6.0-6.5 looks alarming but is expected. The vocabulary has ~10,000 tokens, so a completely random model would score `log(10000) ≈ 9.2`. Going from 9.2 to 5.1 in 2 epochs means the model is genuinely learning — it's picking up that `e4`, `d4`, `Nf3` are common opening moves, that certain move sequences follow certain patterns. Perfect reconstruction isn't the goal; a useful compression is.

**The t-SNE of AE embeddings looked essentially random** — no visible Elo clustering. This is expected and not a failure. The AE was trained to reconstruct move sequences, not to predict Elo. Two games with very different Elo ratings can have similar opening moves and early structure, and the AE correctly puts them close together. What matters is whether adding the 32-d AE vectors improves the main model's MAE, not whether the AE space clusters by Elo on its own.

---

## Model architecture evolution

**v1/v2**: Classes with many `self.layer` attributes, standard PyTorch style. Worked correctly but the architecture was buried under boilerplate.

**v3**: Same structure, more layers added (LayerNorm, GELU, distribution heads).

**v4**: The model is defined with `build_model()` — all the layer components are created as local variables, and the inner `EloPredictor` class simply holds references to them. This keeps the architecture visible in one readable block without the `self.layer1 = ...; self.layer2 = ...` clutter spread across `__init__`. Methods like `encode_sequence`, `game_embedding`, and `probs_to_elo` are named for what they do rather than being mechanical overrides.

The architecture itself:

```
Static branch:
  [board features | clock features | AE embeddings (if enabled)]
  → Linear(n_static → 128) → LayerNorm → GELU → Dropout
  → Linear(128 → 64) → GELU
  → Linear(64 → 32) → GELU

Sequence branch (optional):
  move IDs → Embedding(vocab_size → 64)
  → concat with [normalised_clock | turn_flag]
  → BiLSTM(2 layers, hidden=128, bidirectional)
  → soft attention pooling → 256-d context vector

Fusion:
  concat(sequence_context, static_32d)
  → Linear(seq_dim+32 → 256) → LayerNorm → GELU → Dropout
  → Linear(256 → 128) → GELU → Dropout

Two output heads:
  White: Linear(128 → 40) → Softmax → distribution
  Black: Linear(128 → 40) → Softmax → distribution

Point estimate = Σ (bin_centre × predicted_probability)
```

**Why GELU instead of ReLU?** GELU has a smoother gradient near zero which tends to work better for tabular data and mixed-input architectures. ReLU's hard cutoff at zero can kill gradients for features that are frequently negative (which happens after standardisation).

**Why LayerNorm in the static branch?** The static features, after concatenating board features, clock features, and AE embeddings, come from very different scales even after standardisation. LayerNorm within the MLP helps stabilise activations across the different feature types.

**Why two separate heads?** White and Black share the same trunk — the representation of "what does strong chess look like" is shared. But the heads are separate because White and Black have slightly different predictive signals: White's opening choice reveals more about preparation, Black's response reveals something different. In practice the two MAEs are always within a few points of each other, suggesting the shared trunk is doing most of the work.

---

## Training details

**Loss function**: KL divergence between `log(predicted_distribution)` and the Gaussian target distribution. The `KLDivLoss` expects log-probabilities for the first argument, so we apply `torch.log(probs.clamp(min=1e-9))` — the clamping prevents log(0) from producing -inf.

**Optimizer**: AdamW with weight decay 1e-4. The difference between Adam and AdamW matters here: AdamW applies weight decay directly to the weights rather than through the gradient update, which is the mathematically correct L2 regularisation. For most tasks the difference is small but it's the right choice.

**Learning rate schedule**: Cosine annealing down to 5% of the initial LR. The cosine schedule decays smoothly rather than stepping down, which avoids the situation where a step-based scheduler kills the LR just as the model is about to break through a plateau.

**Weighted random sampler**: The Lichess database skews heavily toward 1400-1800 games. Without resampling, the model sees almost no high-Elo (>2200) or very low-Elo (<1000) games, and its predictions compress toward the majority distribution. The `WeightedRandomSampler` uses inverse-frequency weights per Elo bracket — games in rare brackets get sampled proportionally more. This is why the bracket MAE table shows relatively consistent errors across the 1200-2000 range.

**Gradient clipping** at norm 1.0 is standard for LSTMs and prevents occasional large gradient spikes from derailing training.

**Early stopping**: patience of 3 epochs, requiring at least 0.1% improvement to reset the counter. Without the relative threshold, the early stopping would trigger on tiny fluctuations in val loss. The best checkpoint is saved whenever val loss improves, so even if the model overfits later, the saved weights are from the best epoch.

---

## What the results actually show

The best results across runs:

| Run | Games | MAE (combined) | Within 200 Elo |
|---|---|---|---|
| v1 (broken) | 128 | 1630 | 0% |
| v3 (200k) | 107k | 201 | 57.8% |
| v3 (1M) | 527k | 200 | 58.5% |
| v3 (2M) | 1.05M | 203 | 57.8% |

A few things worth noting:

**200k and 1M games produce almost identical MAE.** This suggests the model is hitting a ceiling from the architecture and features rather than data volume. More data helps with the distribution of ratings seen, but the fundamental difficulty of predicting Elo from a single game without engine evaluations appears to be around 200 MAE.

**The bracket MAE pattern is consistent**: errors are lowest in the 1400-1800 range (~190-195 MAE) and highest at the extremes — around 270-280 for <1000 and 300+ for >2500. This is a known pattern in Elo prediction. Very low-Elo games are erratic and unpredictable (beginners make random mistakes that look like they could be either accidental or tactical), and very high-Elo games look similar to each other to a model without engine evaluations (the difference between a 2400 and a 2600 is mostly in the quality of complex decisions that a move-sequence model can't evaluate without stockfish).

**The overperformer list** (model predicting 1000+ Elo higher than actual) typically contains games where a low-rated player happened to play a coherent, clean game. A 1200-rated player who had a good day and blundered only once will look like a 1700 to a model that can't assess move quality precisely.

**The single-game inference** on the chess.com game (actual: White 1546, Black 1534) predicted around 1750-1780 for both players. This is consistently ~200 points high, which is partly the chess.com vs Lichess rating difference (chess.com ratings tend to run 100-200 points lower for equivalent strength), and partly regression toward the mean in the training data.

---

## Feature importance findings

From the ablation analysis across multiple runs, the most consistent ranking:

1. **game_structure** — consistently the biggest group (total game length, material balance, result). Game length alone is a strong Elo signal: higher-rated players tend to play longer, more technical games. Result contributes because higher-Elo games have more draws.

2. **style** — piece activity, queen moves before move 10, territory depth. The `queen_moves_before_10` feature is a classic beginner tell: lower-rated players move their queen early and get it harassed.

3. **tc_meta** / **clock** — time management is informative. Stronger players use opening time more efficiently (they know the theory), and are less likely to get into severe time trouble.

4. **engine_quality** — when `[%eval]` tags are present this would likely rank first, but in the Lichess 2017 database almost no standard-rated games have annotations, so these features are mostly NaN → 0.

5. **checks** / **castling** — moderate importance. Castle timing is a reliable Elo signal; checks given is more noisy.

6. **captures** — interestingly low. The total number of captures doesn't carry much Elo signal once you control for game length. The timing of the first capture is more informative than the count.

---

## Code style changes from v3 to v4

The main feedback was that the v3 code used too much `self.layer_name = ...` OOP style that felt imported from a different codebase. The v4 changes:

**Vocabulary**: replaced the `Vocab` class with three plain functions (`add_move`, `encode_move`, `vocab_size`) operating on a module-level dict. Does exactly the same thing, no class machinery required.

**Model**: `build_model()` creates all layers as local variables, then defines `EloPredictor` as an inner class that holds references to them. The layers are visible at the top of the function, the forward logic is in methods with descriptive names. The `self.*` explosion in `__init__` is gone.

**Autoencoder**: split into `build_encoder()` and `build_decoder()` factory functions, each using the same inner-class pattern. The AE training loop is flat and readable — create encoder, create decoder, train, encode all games, save.

**Naming throughout**: `move_to_id` instead of `vocab.move_to_id`, `static_cols` selected by `active_prefixes` rather than `enabled_prefixes`, `white_preds/black_preds` instead of `wp_all/bp_all`. Variable names describe what they contain rather than being abbreviations.

**The training loop**: renamed `batch['w_target']` to `batch['white_tgt']` and `batch['b_target']` to `batch['black_tgt']`. Same for `batch['raw_elo']` staying as is — it's clear enough. The batch keys now use full words rather than single-letter abbreviations.

---

## What would actually improve the results further

In rough order of expected impact:

**Engine evaluations** are by far the biggest missing ingredient. The `acpl_*` and `blunder_*` features in `chess_features_final.py` are computed from `[%eval]` tags, but essentially none of the games in the 2017 Lichess database have these. If you could get Stockfish to analyse even 20% of the training games, those features would likely become the top group by a large margin and probably push MAE below 150.

**Longer training with more patience** — the early stopping is currently quite aggressive (patience=3). With the 1M game dataset, the val loss hadn't fully converged at epoch 7-8. Letting it run longer with patience=5-6 might squeeze out another 5-10 MAE points.

**Separating time controls** — a 3-minute blitz game and a 15-minute rapid game are fundamentally different. The model currently handles this by normalising clock features by base time, but training separate models (or adding a time-control embedding) might improve accuracy within each category.

**More AE epochs / larger latent dim** — the AE currently trains for 2 epochs on 15% of data. With patience for a longer run, 5 epochs on 30% of data and a latent dim of 64 might produce better sequence embeddings and improve the main model slightly.

**Position-based features** — the Ouzounis repo passed actual board positions (8×8 arrays) through a CNN rather than just move strings. This is a fundamentally richer representation that can capture piece coordination, pawn structure, and tactical complexity in ways that algebraic move notation cannot. The tradeoff is much larger tensors and significantly slower training, but the ceiling is higher.

---

*v4 notebook, June 2026*
