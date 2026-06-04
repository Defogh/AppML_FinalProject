# Chess Elo Prediction — Project Notes

*Applied Machine Learning project notes. This document explains the full pipeline, the decisions behind each part, and what you should expect when things run.*

---

## What we're actually trying to do

The goal is simple to state but harder to pull off: given a chess game — the moves, maybe the clock times, and some summary statistics — predict the Elo ratings of both players. We predict White and Black in a single forward pass (multi-task learning), which turns out to be better than two separate models because a lot of the signal is shared.

The inspiration for the problem comes from the [Ouzounis repo](https://github.com/HliasOuzounis/Ai-Guess-the-elo), which won 2nd place in a Greek AI competition. Their key insight — which we borrow — is that predicting a full *probability distribution* over Elo ranges is better than predicting a single number. More on that below.

---

## The data pipeline

### Loading from `.pgn.zst`

Lichess releases monthly database dumps as compressed PGN files. We stream-parse them with `zstandard` rather than loading the whole thing into memory. One important thing the v1/v2 runs got wrong: the `TIME_CONTROL = '600+0'` filter was keeping only 10-minute games, which killed 90%+ of the dataset. With 200k games loaded, we were training on about 23k — and with only 128 games surviving in the earliest run, the model was basically averaging to 1600 for every prediction (MAE ~1630). The fix is to either remove the filter or at minimum just set a minimum base time.

### Board features (`chess_features_final.py`)

The feature extractor replays every game on a real python-chess board and computes things that can only be measured by actually following the position:

- **Checks**: how often each player gave check, and the density (checks per move). Higher-Elo players tend to give check when it's actually strong, not as a random harassment tactic.
- **Captures**: when did the first capture happen, pawn captures vs piece captures. Strong players often delay the first capture longer in quiet positions.
- **Castling**: which move did each player castle on, and which side. Lower-Elo players either forget to castle or castle very late.
- **Style signals**: consecutive moves with the same piece (piece activity), queen moves before move 10 (a classic tell for beginners), how deep each player's pieces penetrated into enemy territory.
- **Promotions and en passant**: rare events, but worth tracking.
- **Legal moves at move 5**: a rough measure of opening quality — strong openings tend to leave you with more options.
- **Engine quality** (if the PGN has `[%eval]` annotations): average centipawn loss, inaccuracies, mistakes, blunders, blunder density. This is the strongest single group of features when available, but many games won't have it.

The castle-side columns are categorical (`'king'`, `'queen'`, `'none'`), so they get one-hot encoded before going into the model.

### Clock features

Extracted separately via regex on the `[%clk ...]` tags. Clock time tells you a lot about a player's confidence: how long they spend on average, whether they spend a lot of time in the opening (uncertainty about theory), and whether they end up in time pressure. Everything is normalised by the base time control so a 3-minute game and a 10-minute game are comparable.

### Caching

Board feature extraction is the slowest step — python-chess replays every game move by move. For 200k games this takes several minutes. The notebook caches results to `cache/` as pickle files, keyed by dataset size. If you re-run without changing the data, it just loads from disk.

---

## The sequence autoencoder

This is the main new addition in v3, and it addresses a real problem: the BiLSTM branch adds signal (according to the ablation, about 11% of feature importance when trained properly) but it makes training *very* slow on CPU because we have to run the LSTM on every batch, every epoch.

The autoencoder solves this cleanly. Here's the idea:

**Train the AE once.** The encoder compresses a move sequence (up to 120 moves) into a fixed 32-dimensional vector. The decoder tries to reconstruct the original sequence from that vector. We train it with cross-entropy on the reconstructed tokens, standard teacher forcing.

**Cache the encodings.** After training, we run the encoder on every game in the dataset once and save those 32-d vectors. This takes maybe 2 minutes on CPU.

**Use them as static features.** In the main training loop, instead of running an LSTM every batch, we just concatenate those cached 32-d AE vectors onto the regular static features. Training speed is now essentially the same as the static-only baseline.

**What package do you need?** Nothing extra — the autoencoder is pure PyTorch, which you already have. Just run the AE training cell before flipping `USE_AE = True`.

The architecture is a standard LSTM encoder-decoder:
- **Encoder**: Embedding → BiLSTM (2 layers) → mean-pooling over non-padding positions → linear projection → tanh → 32-d latent vector
- **Decoder**: projects latent vector to LSTM hidden state, then generates tokens autoregressively with teacher forcing during training

The choice of 32 dimensions for the latent space is a reasonable tradeoff — enough to capture opening patterns, tactical tendencies, and game length, but small enough that it doesn't dominate the static features when concatenated.

---

## The soft distribution target (key idea from Ouzounis)

In v1 and v2 we were doing plain regression: the model predicts a single Elo number, and we train with Huber loss against the true Elo. This works, but it ignores something important about chess.

A player rated 1600 doesn't always *play* like a 1600. Some games they're sharp and play 1800-level chess. Other games they miss things a 1400 wouldn't. Their single-game performance is noisy around their true rating. The standard deviation of this noise is roughly 200 Elo points — you can actually derive this from the Elo formula itself (the Ouzounis Decisions_Explained.md has the derivation, which is worth reading).

So instead of training the model to output a single number, we train it to output a **probability distribution over 40 rating bins** (0-4000 split into 100-point buckets). The ground truth for a player rated 1600 is a Gaussian centred at 1600 with σ=200, discretised into those 40 bins. We train with **KL divergence loss** between the predicted and target distributions.

To get a point estimate from the predicted distribution, we take the **weighted mean** of the bin centres — same trick Ouzounis uses. This naturally gives you uncertainty as a side effect: a wide predicted distribution means the model isn't sure, a narrow one means it's confident.

The visual output of this — showing the full distribution for a game rather than just a number — is genuinely interesting to look at and makes for a much better project demonstration.

---

## Model architecture

```
Input
  ├── Static branch:  [board features | clock features | AE embedding (optional)]
  │     Linear(num_static → 128) → LayerNorm → GELU → Dropout
  │     → Linear(128 → 64) → GELU → Linear(64 → 32) → GELU
  │
  └── Sequential branch (optional, USE_SEQUENTIAL=True):
        Embedding(vocab_size → 64) + clock + turn_flag
        → BiLSTM(2 layers, hidden=128, bidirectional)
        → Soft-attention pooling → context vector (256-d)

Fusion:
  concat(context [if seq], static_out)
  → Linear → LayerNorm → GELU → Dropout → Linear → GELU

Two prediction heads:
  White head → Linear(128 → N_BINS) → Softmax → distribution
  Black head → Linear(128 → N_BINS) → Softmax → distribution

Point estimate = Σ (bin_centre × predicted_probability)
```

A few design choices worth mentioning:
- **GELU instead of ReLU** in the static MLP — empirically smoother for tabular data
- **LayerNorm** in the fusion trunk stabilises training, especially when static features and sequence context have different scales
- **Kaiming init** for linear layers, small normal init for embeddings (matching GPT practice)
- **Two separate heads** — sharing the trunk but having separate heads for White and Black lets the model share the "what does good chess look like" representation while still learning that White and Black have slightly different tells (White's opening choice signals more about their preparation level, for example)

---

## Training details

**Optimizer**: AdamW with weight decay 1e-4. AdamW is strictly better than Adam here because the weight decay is applied correctly (decoupled from the gradient scaling).

**Scheduler**: Cosine annealing down to 5% of the initial LR. This works better than ReduceLROnPlateau for this kind of task — the loss often plateaus temporarily before improving again, and step-based schedulers can kill the LR too early.

**Loss**: KL divergence between log-predicted-distribution and target Gaussian distribution (when `USE_DIST_TARGET=True`). Huber loss (δ=150) when doing plain regression.

**Elo-stratified sampler**: High-Elo games (>2000) are rare in the Lichess data. Without resampling, the model sees mostly 1400-1800 games and learns to predict that range well while being terrible at the extremes. We use `WeightedRandomSampler` with inverse-frequency weights per Elo bracket, computed by the `compute_elo_sample_weights` function in `chess_features_final.py`.

**Gradient clipping** at norm 1.0 — standard practice for LSTMs, prevents occasional gradient spikes from derailing training.

**Mixed precision (AMP)**: automatically enabled on CUDA, disabled on CPU. On a GPU this roughly halves memory usage and speeds up training 30-50%.

---

## Why the first runs performed so badly

Looking at the output from `chess_elo_outputs_1e_05_20260603_123942.html`:

- Only **128 games** survived all filters (from 1000 loaded)
- MAE of **1630** on a dataset with mean Elo ~1600 — the model learned exactly nothing, it just predicted the mean
- The attention heatmap showed `White: 1656 (pred 0)` — the model output was literally zero, meaning training collapsed

The root cause was `TIME_CONTROL = '600+0'` combined with loading only 1000 games. Fix: load more games and relax the time control filter. The v3 default sets `TIME_CONTROL = None` and `TC_BASE_MIN = 180` (at least 3-minute games), which typically retains 60-70% of the dataset.

The second problem was the **static-only baseline being used with the sequential model's architecture** — `USE_SEQUENTIAL=False` but the model was still built with LSTM weights that were never trained. This is fine as a baseline, but the first runs were drawing wrong conclusions about model quality.

---

## What the features are actually telling us

From the integrated gradients output (feature importance ranking):

1. `check_density_white` — consistently the strongest single feature. How often you give check relative to total moves is a surprisingly reliable Elo signal.
2. `b_opening_pace` — how much time Black spends per move in the first 5 moves. Stronger players tend to play the opening quickly (they've seen it before).
3. `consec_same_piece_white` — consecutive moves with the same piece. This measures piece activity; higher-rated players tend to have more coordinated piece play rather than moving the same piece repeatedly.
4. `castle_move_white` — move number when White castled. Strong players castle earlier on average.
5. `result_encoded` — whether White won, drew, or lost. This leaks some information (winning games have different character), but it's a legitimate feature since we're predicting historical games.
6. `first_capture_move_white` — when the first capture happened. Positional players delay this; tactical players go for exchanges early.

The engine quality features (`acpl_*`, `blunder_*`) would rank much higher if the dataset had evaluations, but most Lichess games at standard time controls don't have `[%eval]` tags unless they were specifically analysed after the game.

---

## The single-game inference cell

At the bottom of the notebook there's a cell that lets you paste any PGN and get a prediction. A few things to know:

- It uses `chess.pgn.read_game()` to parse the PGN properly, so standard Lichess/chess.com export format will work
- Clock features will be all-NaN if the PGN doesn't have `[%clk ...]` annotations, but the model handles this fine (NaN → 0 after scaling)
- The model checkpoint saves everything needed: the vocab, static column names, scaler parameters, and architecture flags. You can load it in a fresh session without rebuilding the whole pipeline
- If actual Elo is in the PGN headers (`[WhiteElo "..."]`), it shows the comparison

If you're feeding it chess.com games, note that chess.com ratings tend to be about 400 points lower than Lichess for the same player, so predictions may be off — the model was trained on Lichess ratings.

---

## Packages required

Everything uses standard ML packages, plus:

| Package | Why | Install |
|---|---|---|
| `python-chess` | PGN parsing, board replay | `pip install chess` |
| `zstandard` | reading `.pgn.zst` files | `pip install zstandard` |
| `torch` | model training (CPU or GPU) | `pip install torch` |
| `scikit-learn` | StandardScaler, PCA, t-SNE, metrics | `pip install scikit-learn` |
| `joblib` | parallel feature extraction | comes with scikit-learn |
| `pandas`, `numpy`, `matplotlib`, `seaborn` | standard data stack | `pip install pandas numpy matplotlib seaborn` |
| `umap-learn` | UMAP visualisation (optional) | `pip install umap-learn` |
| `captum` | integrated gradients (optional) | `pip install captum` |

The sequence autoencoder uses **no extra packages** — it's pure PyTorch.

---

## Hyperparameter guidance

| Setting | CPU laptop | GPU (mid-range) | Note |
|---|---|---|---|
| `MAX_GAMES` | 50k–200k | 200k–1M | more is always better |
| `EMBED_DIM` | 64 | 128 | embedding dimension |
| `HIDDEN_DIM` | 128 | 256 | LSTM hidden dim |
| `BATCH_SIZE` | 64 | 256 | GPU can go higher |
| `NUM_EPOCHS` | 10–15 | 20–30 | cosine schedule adapts |
| `AE_LATENT_DIM` | 32 | 64 | AE bottleneck |
| `ELO_SIGMA` | 200 | 200 | don't change this |
| `USE_AE` | True | True/False | recommended on CPU |
| `USE_SEQUENTIAL` | False | True | too slow on CPU without AE |

The single most impactful change you can make is loading more games. The model trained on 128 games learned nothing. At 20k+ it starts to learn something. At 200k it should reach around 200-250 MAE for standard ratings, depending on the time control mix.

---

## What "good" performance looks like

For reference, the Ouzounis repo reports 60.5% of predictions within 200 Elo of true rating, trained on 20k games with Stockfish evaluations. Without engine evals:

- **MAE ~250–300 Elo** is achievable on a diverse dataset without engine annotations
- **MAE ~150–200 Elo** is realistic if eval tags are present for most games  
- The model tends to compress toward the mean for extreme ratings (very low and very high Elo) — this is a known limitation of regression-based approaches and is partly why the distribution target helps

The per-bracket MAE breakdown is the most informative diagnostic. If MAE is uniform across brackets, the model is actually generalising. If it's worst at the extremes (as the Ouzounis results showed), you either need more data in those ranges or the stratified sampler isn't working as intended.

---

## A note on what the model can't learn

One thing worth being honest about in a project writeup: predicting Elo from a single game is genuinely hard, even for humans. A 1700-rated player can have a 1400-level game if they're tired, playing a sharp opening they're unfamiliar with, or their opponent makes weird moves that take them out of theory. The model's uncertainty (the width of the predicted distribution) is actually a reasonable reflection of this.

The Ouzounis README captures it well: "a player's strength isn't simply a fixed value but has some variance from game to game." The Gaussian target with σ=200 is deliberately encoding this uncertainty into the training objective.

---

*Last updated: v3 notebook, June 2026*
