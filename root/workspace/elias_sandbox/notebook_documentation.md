# Chess Elo Prediction — Neural Network Documentation

## 1. Overview

This project predicts the Elo ratings of both White and Black players from a single chess game using a neural network. The model combines two complementary information streams:

- **Sequential data**: the move-by-move sequence of a game (move tokens, clock times, turn indicators), processed by a Bidirectional LSTM with an attention mechanism
- **Static data**: 50+ handcrafted game-level features (material balance, capture patterns, castling behaviour, time management), processed by a feed-forward branch

This "Wide & Deep" architecture outputs two simultaneous predictions (White Elo, Black Elo) through a shared backbone with separate prediction heads — a **multi-task learning** setup.

The notebook trains on ~100,000 rated games from the [Lichess database](https://database.lichess.org/) and includes eight post-training analyses ranging from attention visualisation to anomaly detection.

---

## 2. Project Structure

| File | Purpose |
|------|---------|
| `chess_elo_notebook.ipynb` | Main notebook — data loading through all analyses |
| `chess_features_final.py` | Feature extraction module — replays games via `python-chess` |
| `best_chess_model.pth` | Saved model weights (created during training) |
| `*.pgn.zst` | Compressed Lichess PGN data file (input) |

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Neural network training |
| `numpy` | ≥ 1.24 | Numerical operations |
| `pandas` | ≥ 2.0 | Data manipulation |
| `scikit-learn` | ≥ 1.3 | Train/test splits, metrics, PCA, t-SNE |
| `matplotlib` | ≥ 3.7 | Plotting |
| `seaborn` | ≥ 0.12 | Statistical visualisations |
| `python-chess` | ≥ 1.9 | Board replay and move validation |
| `zstandard` | ≥ 0.21 | Decompressing `.pgn.zst` files |
| `joblib` | ≥ 1.3 | Parallel feature extraction |
| `scipy` | ≥ 1.10 | Statistical utilities |

---

## 3. Data Pipeline

### 3.1 Source

Lichess publishes monthly database dumps of all rated games at [database.lichess.org](https://database.lichess.org/). The notebook uses the **May 2017** standard rated dump (`lichess_db_standard_rated_2017-05.pgn.zst`), which contains millions of games compressed with Zstandard.

### 3.2 Loading

The `.pgn.zst` file is stream-decompressed line by line (never fully loaded into memory). For each game, the following PGN headers are extracted:

| Header | Use |
|--------|-----|
| `WhiteElo`, `BlackElo` | Prediction targets |
| `Result` | Encoded as feature (1 / 0 / 0.5) |
| `Moves` | Full move text including clock annotations |
| `TimeControl` | Parsed into `tc_base` (seconds) and `tc_inc` (increment) |
| `ECO` | Opening classification code |
| `Termination` | Used for filtering |

Loading stops after `MAX_GAMES` (default: 100,000) games.

### 3.3 Filtering

Three filters are applied sequentially:

1. **Missing Elo** — games where `WhiteElo` or `BlackElo` is absent or non-numeric are dropped
2. **Termination type** — games ending by `Time forfeit`, `Abandoned`, or `Unterminated` are excluded (these produce unreliable move sequences)
3. **Short games** — games with fewer than `MIN_PLIES` (12) total half-moves are removed (insufficient data for meaningful features)

### 3.4 Time Control Parsing

The `TimeControl` header (e.g., `"600+0"`) is parsed via regex into:

- `tc_base` — base time in seconds (e.g., 600)
- `tc_inc` — increment per move in seconds (e.g., 0)

These are used to normalise all clock-derived features, making them comparable across different time controls.

---

## 4. Feature Engineering

Features come from three sources, totalling approximately 50+ numeric inputs to the model.

### 4.1 Board Features (via `chess_features_final.py`)

Each game is replayed move-by-move on a real chess board using the `python-chess` library. This is the most computationally expensive step (parallelised with `joblib`).

| Group | Features | Description |
|-------|----------|-------------|
| **Structure** | `total_ply_count`, `material_balance_end`, `result_encoded` | Game length, final material difference, outcome |
| **Checks** | `checks_given_white/black`, `check_density_white/black` | How often each side gives check, normalised by game length |
| **Captures** | `first_capture_move_white/black`, `pawn_captures_total`, `piece_captures_total`, `capture_density` | When and how often captures occur |
| **Castling** | `castle_move_white/black`, `castle_side_white/black` | When castling happens and which side (kingside/queenside/none) |
| **Style** | `consec_same_piece_white/black`, `queen_moves_before_10`, `white/black_territory_depth`, `promotions`, `en_passant_captures`, `legal_moves_white/black_move5` | Playing style indicators — piece variety, aggression, mobility |
| **Engine** | `acpl_white/black`, `inaccuracy/mistake/blunder_count_white/black`, `blunder_density_white/black` | Only available when `[%eval]` tags are present in the PGN |

### 4.2 Clock Features

Extracted via regex from the `[%clk H:M:S]` annotations embedded in the move text. All values are **normalised by `tc_base`** so that a player spending 30 seconds in a 10-minute game is comparable to 15 seconds in a 5-minute game.

| Feature | Description |
|---------|-------------|
| `w/b_avg_time_norm` | Average time spent per move (÷ tc_base) |
| `w/b_std_time_norm` | Standard deviation of time per move |
| `w/b_max_time_norm` | Maximum time spent on a single move |
| `w/b_time_pressure` | Number of moves made with < 30 seconds remaining |
| `w/b_opening_pace` | Average time spent on first 5 moves |

Clock features are strong Elo predictors — stronger players tend to have more consistent time usage and spend more time in complex positions.

### 4.3 Move Tokenization

The raw PGN move text is parsed into three parallel sequences:

1. **Move tokens** — SAN notation (e.g., `e4`, `Nf3`, `O-O`) mapped to integer IDs via a vocabulary. `<PAD>` = 0, `<UNK>` = 1, then all observed moves are assigned sequential IDs. Typical vocabulary size: ~1,900 unique moves.
2. **Clock values** — normalised by `tc_base` and clipped to [0, 2]. If fewer clock annotations exist than moves, the remainder is filled with 0.5 (neutral).
3. **Turn indicators** — alternating 0 (White) / 1 (Black) to tell the model which side is moving.

Sequences are capped at `MAX_SEQ_LEN` (120 half-moves = 60 full moves) during batching.

---

## 5. Model Architecture

### 5.1 High-Level Design

```
Move sequence ──→ [Embedding] ──→ [BiLSTM] ──→ [Attention Pooling] ──┐
  + clocks                                                            ├──→ [Shared Layer] ──→ White Elo head
  + turns                                                             │                  └──→ Black Elo head
                                                                      │
Static features (50+) ──→ [Feed-Forward Branch] ─────────────────────┘
```

This is a **Wide & Deep** architecture:
- **Deep path**: sequential move data → BiLSTM → attention → context vector (256-d)
- **Wide path**: static game features → two dense layers → feature vector (32-d)
- **Merge**: concatenate (288-d) → shared dense layer (128-d) → two prediction heads

### 5.2 Layer Details

| Component | Specification |
|-----------|---------------|
| **Move Embedding** | `Embedding(vocab_size, 64, padding_idx=0)` |
| **LSTM Input** | Concatenation of embedding (64) + clock (1) + turn (1) = **66** dimensions |
| **BiLSTM** | 2 layers, hidden_dim=128, bidirectional → output 256 per timestep |
| **Attention** | `Linear(256→64) → Tanh → Linear(64→1)` → softmax over sequence |
| **Static Branch** | `Linear(N→64) → ReLU → Dropout(0.3) → Linear(64→32) → ReLU` |
| **Shared Trunk** | `Linear(288→128) → ReLU → Dropout(0.3)` |
| **White Head** | `Linear(128→1)` |
| **Black Head** | `Linear(128→1)` |

### 5.3 Attention Mechanism

Rather than using only the final LSTM hidden state (which compresses the entire game into a single vector and loses information about early moves), the attention mechanism computes a **weighted average** over all LSTM timestep outputs:

1. Each timestep's 256-d output is projected to a scalar "importance score"
2. Padding positions are masked to −∞ before softmax
3. Softmax produces normalised weights across all real moves
4. The context vector is the weighted sum of all timestep outputs

This lets the model learn that, for example, a blunder on move 25 might be more informative about a player's rating than a standard opening move on move 3.

### 5.4 Multi-Task Setup

The model predicts **both White and Black Elo simultaneously** from the same game. The shared BiLSTM and trunk must learn representations useful for predicting *either* colour, which:

- **Acts as regularisation** — prevents overfitting to one side's patterns
- **Doubles the effective training signal** — each game provides two gradient updates
- **Forces colour-aware learning** — the turn indicator + separate heads let the model distinguish White vs Black behaviour

---

## 6. Design Considerations

### Why BiLSTM over unidirectional LSTM?

A unidirectional LSTM processes moves left-to-right, meaning the representation of move 5 has no knowledge of what happens on move 40. In chess, later moves (especially blunders or endgame technique) are highly informative about earlier decisions. The backward pass of a BiLSTM lets every timestep incorporate information from the entire game.

### Why attention over final-hidden-state pooling?

The final hidden state of an LSTM is biased towards the end of the sequence. In chess, critical moments (a sacrifice, a missed tactic, time trouble) can happen anywhere. Attention lets the model dynamically weight each move's importance, and the learned weights are **interpretable** — we can visualise which moves the model considered most revealing.

### Why Huber loss (δ=100) over MSE or KL divergence?

| Loss | Problem |
|------|---------|
| **MSE** | Squares the error, so a single 800-Elo outlier dominates the gradient. Elo data has heavy tails. |
| **MAE** | Not differentiable at zero, noisy gradients |
| **KL divergence on Elo buckets** | Requires converting Elo to a probability distribution over bins. Adds complexity (choosing bin width, smoothing) without clear benefit for point estimation. |
| **Huber (δ=100)** | Behaves like MSE for errors < 100 Elo (smooth, efficient) and like MAE for errors > 100 (robust to outliers). Best of both worlds. |

### Why gradient clipping (max_norm=1.0)?

LSTMs are prone to exploding gradients, especially with bidirectional architectures and long sequences. Clipping the global gradient norm to 1.0 prevents training instability without altering the gradient direction.

### Why Xavier initialisation?

The default PyTorch initialisation (Kaiming uniform) is designed for ReLU activations. Xavier uniform is better suited for the mixed activation landscape (Tanh in attention, ReLU in dense layers) and tends to produce faster convergence in sequence models.

### Why AdamW over Adam?

AdamW decouples weight decay from the gradient update, providing more consistent regularisation. With `weight_decay=1e-4`, it gently penalises large weights without interfering with the adaptive learning rate.

---

## 7. Training Procedure

| Setting | Value |
|---------|-------|
| Optimiser | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Loss | HuberLoss(δ=100) — summed for both heads |
| Gradient clipping | max_norm=1.0 |
| Batch size | 128 |
| Max epochs | 15 |
| Early stopping | Patience = 3 epochs without validation improvement |
| Best model saving | `best_chess_model.pth` — saved on each validation improvement |

### Training/Validation/Test Split

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | 80% | Model parameter updates |
| Validation | 10% | Hyperparameter tuning, early stopping, learning rate scheduling |
| Test | 10% | Final unbiased evaluation (never seen during training) |

The total loss per batch is: `L = Huber(ŷ_white, y_white) + Huber(ŷ_black, y_black)`

Both heads contribute equally to the gradient.

---

## 8. Evaluation Metrics

### 8.1 Primary Metrics

| Metric | What it measures |
|--------|-----------------|
| **MAE** (Mean Absolute Error) | Average magnitude of prediction errors in Elo points. Reported separately for White and Black. |
| **Median AE** | More robust to outliers than MAE — the "typical" error. |
| **Within 100 Elo** | Percentage of predictions within ±100 of the true rating — a practical accuracy measure. |
| **Within 200 Elo** | Looser threshold — what fraction of games does the model get "roughly right"? |

### 8.2 Per-Bracket MAE

Predictions are binned by the player's true Elo:

| Bracket | Typical behaviour |
|---------|-------------------|
| 0–1000 | Higher MAE — fewer training samples, more erratic play |
| 1200–1800 | Lowest MAE — most data, most "typical" play patterns |
| 2000–2500 | Moderate MAE — less data, but more consistent play |
| 2500+ | Highest MAE — very few samples, elite play is subtle |

This analysis reveals **where the model is reliable** and where it struggles.

---

## 9. Analyses and Visualisations

The notebook includes eight post-training analyses:

### 9.1 Predicted vs Actual Scatter Plots

2×2 grid showing White and Black Elo predictions against ground truth. Points near the red diagonal indicate accurate predictions. Systematic deviations (e.g., clustering above the diagonal for low Elo) reveal bias.

### 9.2 Error Distribution

Histograms of (predicted − actual) for each colour with KDE overlay. A symmetric distribution centred at zero indicates an unbiased model. Skew or fat tails indicate systematic over/under-prediction.

### 9.3 Per-Bracket MAE Bar Chart

Horizontal bar chart showing MAE within each Elo bracket. Immediately reveals which rating ranges the model handles well and which it struggles with. Accompanied by a printed table with sample counts.

### 9.4 Attention Heatmap

For a single game (30–50 moves), the attention weights are visualised as a colour-coded heatmap over the move sequence. Move labels are coloured by side (black text = White's move, gray = Black's move). This is the most **interpretable** output — it shows which specific moves the model considers most informative about a player's rating.

### 9.5 Progressive Elo Estimation

Six games of varying true Elo are truncated at cutpoints (5, 10, 15, 20, 30, 40, 50, full). At each truncation, the model predicts the Elo. The resulting convergence curves show **how quickly the model identifies a player's level**. Typical finding: the model needs ~15–20 moves to stabilise, with the biggest uncertainty reduction happening in the opening.

### 9.6 Game-Level t-SNE Clustering

The 128-dimensional shared representation (from `get_game_vector()`) is projected to 2D via t-SNE for ~3,000 test games. Two views:

- **All games** coloured by average Elo (coolwarm colourmap) — reveals whether the latent space organises games by skill level
- **Extremes only** (< 1300 vs > 2000) — tests whether the model clearly separates beginner and expert games

### 9.7 Move Embedding PCA

The learned 64-dimensional move embeddings are projected to 2D via PCA. Three move categories are highlighted:

- **Common openings** (e4, d4, Nf3, etc.) — expected to cluster
- **Castling** (O-O, O-O-O) — structurally distinct moves
- **Rare/edge moves** (h4, Na3, etc.) — expected to appear on the periphery

This visualisation shows what the model has learned about the **semantic relationships** between chess moves purely from Elo prediction.

### 9.8 Anomaly Detection

Games where the model's prediction diverges significantly from reality (|error| > 400 Elo) are flagged. Two categories:

- **Overperformers** (predicted ≫ actual) — the player performed far above their rating. Potential explanations: engine assistance, smurf account, or a genuinely brilliant game
- **Underperformers** (predicted ≪ actual) — the player performed far below their rating. Potential explanations: tilt, intoxication, time trouble, or intentional sandbagging

### 9.9 Opening Analysis

If ECO codes are available, games are grouped by opening family (A–E). A grouped bar chart compares average actual Elo vs average predicted Elo per family. Additionally, the top 5 and bottom 5 specific ECO codes by average Elo are printed, revealing which openings are favoured at different skill levels.

---

## 10. Hyperparameter Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MAX_GAMES` | 100,000 | Balance between training data volume and computation time |
| `BATCH_SIZE` | 128 | Standard for sequence models; fits comfortably in GPU memory |
| `EMBED_DIM` | 64 | Sufficient to capture move semantics; ~1,900 vocab items |
| `HIDDEN_DIM` | 128 | BiLSTM outputs 256 per step — expressive without being wasteful |
| `NUM_EPOCHS` | 15 | With early stopping (patience=3), typically converges in 8–12 |
| `LR` | 1e-3 | Standard starting point for AdamW |
| `DROPOUT` | 0.3 | Applied in static branch and shared trunk — prevents overfitting |
| `MIN_PLIES` | 12 | Games shorter than 6 full moves provide insufficient signal |
| `MAX_SEQ_LEN` | 120 | 60 full moves covers >95% of games; longer sequences add noise |
| `VAL_FRAC` / `TEST_FRAC` | 0.10 / 0.10 | Standard 80/10/10 split |
| `Huber δ` | 100 | Transition point from MSE to MAE behaviour at 100 Elo error |

---

## 11. How to Run

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn python-chess zstandard joblib scipy
```

### Steps

1. Place `chess_elo_notebook.ipynb` and `chess_features_final.py` in the same directory
2. Ensure the data file path in `DATA_PATH` points to your `.pgn.zst` file
3. Open in VS Code or JupyterLab and run cells sequentially
4. Training takes approximately 15–30 minutes on a modern CPU, 3–5 minutes with a CUDA GPU

### Expected Output

- Console prints at each stage (data loading, feature extraction, training epochs)
- `best_chess_model.pth` saved in the working directory
- 8 matplotlib figures generated inline
- Printed tables for metrics, anomalies, and opening analysis

---

## 12. Potential Improvements and Future Work

| Improvement | Expected Impact | Complexity |
|-------------|----------------|------------|
| **Transformer encoder** instead of LSTM | Better long-range dependencies, parallelisable | High |
| **Piece-square decomposition** | Instead of one token per SAN move, decompose into (piece, from_sq, to_sq) — gives structural board understanding | Medium |
| **Data augmentation via colour swap** | Mirror games (swap White/Black) to double effective dataset | Low |
| **Pre-training on move prediction** | Train the embedding + LSTM to predict the next move first, then fine-tune for Elo | High |
| **Elo-conditioned generation** | Reverse the model — given an Elo, generate move sequences typical of that level | High |
| **Cross-time-control analysis** | Train on blitz, test on classical (or vice versa) — how transferable are skill signatures? | Low |
| **Sample weighting by Elo** | Upweight rare high/low Elo games using `compute_elo_sample_weights()` from `chess_features_final.py` | Low |
| **Ensemble with gradient boosting** | Combine NN predictions with LightGBM on static features for a potential MAE reduction | Medium |
