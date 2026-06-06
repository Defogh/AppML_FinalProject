"""build_notebooks.py — generates both chess notebooks."""
import os

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK 1 — chess_playstyle_clustering.ipynb
# ═══════════════════════════════════════════════════════════════════════════════

nb1_cells = []

def md(src): return new_markdown_cell(src)
def code(src): return new_code_cell(src)

nb1_cells += [
md("""# Chess Play-Style Clustering
Unsupervised exploration of player archetypes using board features + clock features.

**Pipeline**
1. Feature engineering (same loader as `chess_elo_v_final`)
2. Per-player profile aggregation
3. Dimensionality reduction — t-SNE and UMAP
4. Clustering — K-Means and HDBSCAN
5. Archetype labelling & visualisation

> Toggle every behaviour from the **CONFIGURATION** cell — nowhere else.
"""),

code("""\
# pip install chess zstandard lightgbm umap-learn hdbscan pandas scikit-learn matplotlib seaborn joblib
import io, re, time, os, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import zstandard as zstd

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from joblib import Parallel, delayed

import umap
import hdbscan

from chess_features_final import extract_features_dataframe

warnings.filterwarnings('ignore')
print('All imports OK')
"""),

md("## ⚙️ CONFIGURATION — edit here only"),

code("""\
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "../../data/villads_data/lichess_db_standard_rated_2017-11.pgn.zst"
MAX_GAMES    = 500_000      # reduce for faster iteration
RANDOM_STATE = 42
N_JOBS       = -1

# ── filters ───────────────────────────────────────────────────────────────────
FILTER_BASE_SECONDS  = 600   # None = all time controls
MIN_TOTAL_PLIES      = 12
EXCLUDE_TERMINATIONS = {"Time forfeit", "Abandoned", "Unterminated"}

# ── feature groups ────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    'structure':  True,
    'checks':     True,
    'captures':   True,
    'castling':   True,
    'style':      True,
    'clock':      True,
    'engine':     False,   # needs [%eval] tags in PGN
}

# ── per-player aggregation ────────────────────────────────────────────────────
# Each player appears in multiple games; we summarise their stats.
# Set False to use raw per-game rows (much larger matrix).
AGGREGATE_BY_PLAYER  = False   # True requires 'Username' column or similar
MIN_GAMES_PER_PLAYER = 5       # only used when AGGREGATE_BY_PLAYER=True

# ── dimensionality reduction ──────────────────────────────────────────────────
# t-SNE
TSNE_PERPLEXITY   = 40
TSNE_N_ITER       = 1000
TSNE_SAMPLE       = 30_000   # subsample for speed (None = all)

# UMAP
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.05
UMAP_METRIC       = 'euclidean'

# ── clustering ────────────────────────────────────────────────────────────────
# K-Means
KMEANS_K          = 7        # number of clusters
RUN_ELBOW         = True     # plot elbow curve to help choose K
ELBOW_K_RANGE     = range(2, 14)

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE  = 500
HDBSCAN_MIN_SAMPLES       = 50
HDBSCAN_CLUSTER_SELECTION = 'eom'  # 'eom' or 'leaf'

# ── archetype labels ──────────────────────────────────────────────────────────
# Used to annotate cluster scatter plots after you've inspected the centroids.
# Keys are 0-indexed cluster IDs; leave empty and the plots use numeric IDs.
KMEANS_LABELS = {
    # 0: "Aggressive Attacker",
    # 1: "Blunder King",
    # 2: "Fast & Loose",
    # 3: "Positional Player",
    # 4: "Slow & Steady",
    # 5: "Endgame Specialist",
    # 6: "Solid Defender",
}

# ── output ────────────────────────────────────────────────────────────────────
VERBOSE_PLOTS  = True
SAVE_CSV       = True
OUTPUT_CSV     = f'chess_clusters_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
"""),

md("## 1 · Load data"),

code("""\
games_list = []
dctx = zstd.ZstdDecompressor()

t0 = time.time()
with open(DATA_PATH, 'rb') as compressed_file:
    with dctx.stream_reader(compressed_file) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        current_game = {}
        for line in text_stream:
            line = line.strip()
            if line.startswith('['):
                tag = line.split(' ')[0][1:]
                val = line.split('"')[1]
                if tag in ('White', 'Black', 'WhiteElo', 'BlackElo',
                           'TimeControl', 'ECO', 'Termination', 'Result'):
                    current_game[tag] = val
            elif line.startswith('1.'):
                current_game['Moves'] = line
                if 'WhiteElo' in current_game and 'BlackElo' in current_game:
                    games_list.append(current_game)
                current_game = {}
                if len(games_list) >= MAX_GAMES:
                    break

df_raw = pd.DataFrame(games_list)
df_raw['WhiteElo'] = pd.to_numeric(df_raw['WhiteElo'], errors='coerce')
df_raw['BlackElo'] = pd.to_numeric(df_raw['BlackElo'], errors='coerce')
df_raw = df_raw.dropna(subset=['WhiteElo', 'BlackElo']).copy()
df_raw['WhiteElo'] = df_raw['WhiteElo'].astype(int)
df_raw['BlackElo'] = df_raw['BlackElo'].astype(int)
df_raw['Moves']    = df_raw['Moves'].fillna('').astype(str)

print(f'Loaded {len(df_raw):,} games in {time.time()-t0:.1f}s')

# ── optional time-control filter ──────────────────────────────────────────────
if FILTER_BASE_SECONDS is not None:
    _tc_base = (
        df_raw['TimeControl'].astype(str)
        .str.extract(r'^(\\d+)\\+')[0]
        .astype(float)
    )
    df_raw = df_raw[_tc_base == FILTER_BASE_SECONDS].copy().reset_index(drop=True)
    print(f'After TC filter ({FILTER_BASE_SECONDS}s): {len(df_raw):,} games')

if EXCLUDE_TERMINATIONS and 'Termination' in df_raw.columns:
    before = len(df_raw)
    df_raw = df_raw[~df_raw['Termination'].isin(EXCLUDE_TERMINATIONS)].copy().reset_index(drop=True)
    print(f'After termination filter: {len(df_raw):,} games (dropped {before-len(df_raw):,})')

# ── numeric time-control features ─────────────────────────────────────────────
if 'TimeControl' in df_raw.columns:
    _tc = df_raw['TimeControl'].astype(str).str.extract(r'^(\\d+)\\+(\\d+)')
    df_raw['tc_base']      = pd.to_numeric(_tc[0], errors='coerce').fillna(0).astype(float)
    df_raw['tc_increment'] = pd.to_numeric(_tc[1], errors='coerce').fillna(0).astype(float)

df_raw.head(3)
"""),

md("## 2 · Clock features"),

code("""\
_CLK_RE = re.compile(r'\\[%clk\\s+(\\d+):(\\d+):(\\d+)\\]')

def parse_clock_features(moves_string: str, time_control: str = '?') -> dict:
    clocks = [int(h)*3600 + int(m)*60 + int(s)
              for h, m, s in _CLK_RE.findall(moves_string)]
    clk_w, clk_b = clocks[0::2], clocks[1::2]

    tc_m      = re.match(r'(\\d+)\\+(\\d+)', str(time_control))
    base      = int(tc_m.group(1)) if tc_m else None
    increment = int(tc_m.group(2)) if tc_m else 0
    norm      = base if base else 1

    def _spent(seq, start):
        out, prev = [], start
        for c in seq:
            if prev is not None:
                s = prev - c + increment
                if s >= 0: out.append(s)
            prev = c
        return out

    sw = _spent(clk_w, base)
    sb = _spent(clk_b, base)

    def _stats(seq):
        if not seq: return 0., 0., 0.
        a = np.array(seq)
        return float(a.mean()), float(a.std()), float(a.max())

    def _trend(seq):
        if len(seq) < 2: return 0.0
        x = np.arange(len(seq))
        slope = np.polyfit(x, np.array(seq, dtype=float), 1)[0]
        return float(slope)

    def _window_mean(seq, n=10, which='first'):
        if not seq: return 0.0
        return float(np.mean(seq[:n])) if which == 'first' else float(np.mean(seq[-n:]))

    aw, sw2, mw = _stats(sw)
    ab, sb2, mb = _stats(sb)

    return {
        'avg_time_norm_white':         aw / norm,
        'std_time_norm_white':         sw2 / norm,
        'max_time_norm_white':         mw / norm,
        'time_pressure_white':         sum(1 for c in clk_w if c < 10),
        'opening_pace_norm_white':     (np.mean(sw[:10]) if sw else 0.) / norm,
        'clock_remaining_norm_white':  (clk_w[-1] / norm) if clk_w else 0.,
        'opening_time_norm_white':     _window_mean(sw, 10, 'first') / norm,
        'endgame_time_norm_white':     _window_mean(sw, 10, 'last') / norm,
        'time_spent_trend_norm_white': _trend(sw) / norm,
        'avg_time_norm_black':         ab / norm,
        'std_time_norm_black':         sb2 / norm,
        'max_time_norm_black':         mb / norm,
        'time_pressure_black':         sum(1 for c in clk_b if c < 10),
        'opening_pace_norm_black':     (np.mean(sb[:10]) if sb else 0.) / norm,
        'clock_remaining_norm_black':  (clk_b[-1] / norm) if clk_b else 0.,
        'opening_time_norm_black':     _window_mean(sb, 10, 'first') / norm,
        'endgame_time_norm_black':     _window_mean(sb, 10, 'last') / norm,
        'time_spent_trend_norm_black': _trend(sb) / norm,
    }

if FEATURE_GROUPS['clock']:
    tc_col = df_raw['TimeControl'] if 'TimeControl' in df_raw.columns else ['?']*len(df_raw)
    clock_records = Parallel(n_jobs=N_JOBS)(
        delayed(parse_clock_features)(m, t)
        for m, t in zip(df_raw['Moves'], tc_col)
    )
    df_clocks = pd.DataFrame(clock_records, index=df_raw.index)
    print('Clock features:', df_clocks.shape)
else:
    df_clocks = pd.DataFrame(index=df_raw.index)
    print('Clock features skipped')
"""),

md("## 3 · Board features"),

code("""\
t0       = time.time()
df_feats = extract_features_dataframe(df_raw, n_jobs=N_JOBS)
elapsed  = time.time() - t0
print(f'Extracted {len(df_feats):,} games in {elapsed:.1f}s  ({elapsed/len(df_feats)*1000:.1f} ms/game)')
"""),

md("## 4 · Build clustering feature matrix"),

code("""\
# ── feature lists ─────────────────────────────────────────────────────────────
_BOARD_FEATURES = {
    'structure': ['total_ply_count', 'material_balance_end', 'result_encoded'],
    'checks':    ['checks_given_white', 'checks_given_black',
                  'check_density_white', 'check_density_black'],
    'captures':  ['first_capture_move_white', 'first_capture_move_black',
                  'pawn_captures_total', 'piece_captures_total', 'capture_density'],
    'castling':  ['castle_move_white', 'castle_move_black'],
    'style':     ['consec_same_piece_white', 'consec_same_piece_black',
                  'queen_moves_before_10', 'white_territory_depth', 'black_territory_depth',
                  'promotions', 'en_passant_captures',
                  'legal_moves_white_move5', 'legal_moves_black_move5'],
    'engine':    ['acpl_white', 'inaccuracy_count_white', 'mistake_count_white',
                  'blunder_count_white', 'blunder_density_white',
                  'acpl_black', 'inaccuracy_count_black', 'mistake_count_black',
                  'blunder_count_black', 'blunder_density_black'],
}
_CLOCK_FEATURES = [
    'avg_time_norm_white', 'std_time_norm_white', 'max_time_norm_white',
    'time_pressure_white', 'opening_pace_norm_white', 'clock_remaining_norm_white',
    'opening_time_norm_white', 'endgame_time_norm_white', 'time_spent_trend_norm_white',
    'avg_time_norm_black', 'std_time_norm_black', 'max_time_norm_black',
    'time_pressure_black', 'opening_pace_norm_black', 'clock_remaining_norm_black',
    'opening_time_norm_black', 'endgame_time_norm_black', 'time_spent_trend_norm_black',
]

CLUSTER_FEATURES = []
for grp, cols in _BOARD_FEATURES.items():
    if FEATURE_GROUPS.get(grp, False):
        CLUSTER_FEATURES += cols
if FEATURE_GROUPS['clock']:
    CLUSTER_FEATURES += _CLOCK_FEATURES

# ── derived play-style composites ─────────────────────────────────────────────
# These are extra engineered features specifically useful for clustering
# player archetypes — they summarise cross-side patterns.

meta_cols = ['WhiteElo', 'BlackElo']
if 'tc_base' in df_raw.columns:
    meta_cols += ['tc_base', 'tc_increment']

df = (
    df_raw[meta_cols]
    .join(df_feats, how='inner')
    .join(df_clocks, how='inner')
)
df = df.replace([np.inf, -np.inf], np.nan)

# Symmetric (colour-blind) composites
df['avg_elo']             = (df['WhiteElo'] + df['BlackElo']) / 2
df['check_density_avg']   = df[['check_density_white','check_density_black']].mean(axis=1)
df['capture_agression']   = df['piece_captures_total'] / (df['total_ply_count'].clip(1))
df['pawn_aggression']     = df['pawn_captures_total']  / (df['total_ply_count'].clip(1))
df['consec_piece_avg']    = df[['consec_same_piece_white','consec_same_piece_black']].mean(axis=1)
df['time_pressure_avg']   = df[['time_pressure_white','time_pressure_black']].mean(axis=1)
df['avg_time_avg']        = df[['avg_time_norm_white','avg_time_norm_black']].mean(axis=1)
df['clock_remaining_avg'] = df[['clock_remaining_norm_white','clock_remaining_norm_black']].mean(axis=1)
df['opening_pace_avg']    = df[['opening_pace_norm_white','opening_pace_norm_black']].mean(axis=1)
df['time_variability']    = df[['std_time_norm_white','std_time_norm_black']].mean(axis=1)

COMPOSITE_FEATURES = [
    'avg_elo',
    'check_density_avg', 'capture_agression', 'pawn_aggression',
    'consec_piece_avg', 'queen_moves_before_10',
    'time_pressure_avg', 'avg_time_avg', 'clock_remaining_avg',
    'opening_pace_avg', 'time_variability',
    'total_ply_count', 'castle_move_white', 'castle_move_black',
    'white_territory_depth', 'black_territory_depth',
    'promotions', 'en_passant_captures',
    'material_balance_end',
]
if FEATURE_GROUPS['engine']:
    COMPOSITE_FEATURES += [
        'blunder_density_white', 'blunder_density_black',
        'acpl_white', 'acpl_black',
    ]

COMPOSITE_FEATURES = [c for c in COMPOSITE_FEATURES if c in df.columns]
print(f'Composite clustering features: {len(COMPOSITE_FEATURES)}')

# ── impute + scale ─────────────────────────────────────────────────────────────
X_raw = df[COMPOSITE_FEATURES].copy()
imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(X_raw)
scaler  = RobustScaler()
X_scaled = scaler.fit_transform(X_imp)

print(f'Clustering matrix: {X_scaled.shape}')
"""),

md("## 5 · K-Means — elbow + fit"),

code("""\
if RUN_ELBOW:
    inertias, sil_scores = [], []
    for k in ELBOW_K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        # silhouette on a subsample to keep it fast
        _idx = np.random.default_rng(RANDOM_STATE).choice(len(X_scaled),
               size=min(5000, len(X_scaled)), replace=False)
        sil_scores.append(silhouette_score(X_scaled[_idx], labels[_idx]))
        print(f'  k={k}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}')

    if VERBOSE_PLOTS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        ax1.plot(list(ELBOW_K_RANGE), inertias, 'o-', color='#4e91d9')
        ax1.set_xlabel('k'); ax1.set_ylabel('Inertia')
        ax1.set_title('K-Means Elbow Curve'); ax1.grid(alpha=0.3)

        ax2.plot(list(ELBOW_K_RANGE), sil_scores, 'o-', color='#e05252')
        ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette score')
        ax2.set_title('Silhouette Score vs k'); ax2.grid(alpha=0.3)

        plt.tight_layout(); plt.show()
        print(f'Best silhouette at k={list(ELBOW_K_RANGE)[np.argmax(sil_scores)]}')

# ── final K-Means fit ─────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE, n_init=20)
df['kmeans_cluster'] = km_final.fit_predict(X_scaled)
print(f'K-Means (k={KMEANS_K}) cluster sizes:')
print(df['kmeans_cluster'].value_counts().sort_index())
"""),

md("## 6 · HDBSCAN"),

code("""\
# HDBSCAN needs a lower-dimensional space — we use UMAP first (faster + better structure)
# so run UMAP before HDBSCAN.
print('Running UMAP for HDBSCAN pre-reduction...')
t0 = time.time()
reducer_hdb = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=0.0,          # 0 = tighter clusters, better for HDBSCAN
    n_components=10,       # 10-D embedding fed to HDBSCAN
    metric=UMAP_METRIC,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    low_memory=False,
)
X_umap_hdb = reducer_hdb.fit_transform(X_scaled)
print(f'UMAP (10-D) done in {time.time()-t0:.1f}s')

print('Running HDBSCAN...')
t0 = time.time()
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    cluster_selection_method=HDBSCAN_CLUSTER_SELECTION,
    metric='euclidean',
    core_dist_n_jobs=N_JOBS,
    prediction_data=True,
)
df['hdbscan_cluster'] = clusterer.fit_predict(X_umap_hdb)
n_found   = df['hdbscan_cluster'].nunique() - (1 if -1 in df['hdbscan_cluster'].values else 0)
n_noise   = (df['hdbscan_cluster'] == -1).sum()
print(f'HDBSCAN done in {time.time()-t0:.1f}s  |  clusters: {n_found}  noise: {n_noise:,}')
print(df['hdbscan_cluster'].value_counts().sort_index())
"""),

md("## 7 · t-SNE embedding"),

code("""\
print('Running t-SNE...')
t0 = time.time()

# Subsample if configured
if TSNE_SAMPLE and TSNE_SAMPLE < len(X_scaled):
    rng     = np.random.default_rng(RANDOM_STATE)
    ts_idx  = rng.choice(len(X_scaled), size=TSNE_SAMPLE, replace=False)
    X_tsne_input = X_scaled[ts_idx]
    df_tsne = df.iloc[ts_idx].copy().reset_index(drop=True)
    _subset = True
else:
    X_tsne_input = X_scaled
    df_tsne = df.copy().reset_index(drop=True)
    ts_idx  = np.arange(len(X_scaled))
    _subset = False

# Use PCA initialisation for reproducibility & speed
tsne = TSNE(
    n_components=2,
    perplexity=TSNE_PERPLEXITY,
    n_iter=TSNE_N_ITER,
    random_state=RANDOM_STATE,
    init='pca',
    learning_rate='auto',
    n_jobs=N_JOBS,
)
emb_tsne = tsne.fit_transform(X_tsne_input)
print(f't-SNE done in {time.time()-t0:.1f}s')
"""),

md("## 8 · UMAP embedding (2-D visualisation)"),

code("""\
print('Running UMAP (2-D)...')
t0 = time.time()

reducer_2d = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=2,
    metric=UMAP_METRIC,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    low_memory=False,
)
emb_umap = reducer_2d.fit_transform(X_scaled)
print(f'UMAP (2-D) done in {time.time()-t0:.1f}s')

df['umap_x'] = emb_umap[:, 0]
df['umap_y'] = emb_umap[:, 1]
"""),

md("## 9 · Visualise — t-SNE vs UMAP, coloured by cluster & Elo"),

code("""\
def _label(cluster_id: int, label_map: dict) -> str:
    return label_map.get(cluster_id, f'Cluster {cluster_id}')

# ── palette ───────────────────────────────────────────────────────────────────
PALETTE = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#bfef45','#fabed4','#469990',
    '#dcbeff','#9a6324','#fffac8','#800000','#aaffc3',
]

def scatter_clusters(ax, x, y, labels, label_map, title, alpha=0.25, s=4):
    unique = sorted(set(labels))
    for i, c in enumerate(unique):
        mask  = labels == c
        color = '#aaaaaa' if c == -1 else PALETTE[i % len(PALETTE)]
        name  = 'Noise' if c == -1 else _label(c, label_map)
        ax.scatter(x[mask], y[mask], c=color, s=s, alpha=alpha,
                   rasterized=True, label=f'{name} (n={mask.sum():,})')
    ax.legend(loc='upper right', fontsize=7, markerscale=3)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

# ── t-SNE plots ───────────────────────────────────────────────────────────────
if VERBOSE_PLOTS:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # K-Means on t-SNE
    scatter_clusters(axes[0],
                     emb_tsne[:,0], emb_tsne[:,1],
                     df_tsne['kmeans_cluster'].values,
                     KMEANS_LABELS, f't-SNE — K-Means (k={KMEANS_K})')

    # HDBSCAN on t-SNE
    scatter_clusters(axes[1],
                     emb_tsne[:,0], emb_tsne[:,1],
                     df_tsne['hdbscan_cluster'].values,
                     {}, 't-SNE — HDBSCAN')

    # Elo on t-SNE
    sc = axes[2].scatter(emb_tsne[:,0], emb_tsne[:,1],
                         c=df_tsne['avg_elo'].values,
                         cmap='RdYlGn', s=4, alpha=0.2, rasterized=True)
    plt.colorbar(sc, ax=axes[2], label='Avg Elo')
    axes[2].set_title('t-SNE — Avg Elo'); axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.suptitle('t-SNE Embeddings', fontsize=14, y=1.01)
    plt.tight_layout(); plt.show()

    # ── UMAP plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    scatter_clusters(axes[0],
                     df['umap_x'].values, df['umap_y'].values,
                     df['kmeans_cluster'].values,
                     KMEANS_LABELS, f'UMAP — K-Means (k={KMEANS_K})')

    scatter_clusters(axes[1],
                     df['umap_x'].values, df['umap_y'].values,
                     df['hdbscan_cluster'].values,
                     {}, 'UMAP — HDBSCAN')

    sc = axes[2].scatter(df['umap_x'].values, df['umap_y'].values,
                         c=df['avg_elo'].values,
                         cmap='RdYlGn', s=4, alpha=0.2, rasterized=True)
    plt.colorbar(sc, ax=axes[2], label='Avg Elo')
    axes[2].set_title('UMAP — Avg Elo'); axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.suptitle('UMAP Embeddings', fontsize=14, y=1.01)
    plt.tight_layout(); plt.show()
"""),

md("## 10 · Cluster profiling — centroids & radar charts"),

code("""\
# ── centroid table ────────────────────────────────────────────────────────────
_profile_cols = [
    'avg_elo', 'check_density_avg', 'capture_agression',
    'time_pressure_avg', 'avg_time_avg', 'clock_remaining_avg',
    'opening_pace_avg', 'time_variability',
    'queen_moves_before_10', 'consec_piece_avg',
    'total_ply_count', 'promotions', 'en_passant_captures',
]
_profile_cols = [c for c in _profile_cols if c in df.columns]

def cluster_profile(df_in: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    grp = df_in.groupby(cluster_col)[_profile_cols]
    return grp.mean().round(3)

km_profile  = cluster_profile(df, 'kmeans_cluster')
hdb_profile = cluster_profile(df[df['hdbscan_cluster'] >= 0], 'hdbscan_cluster')

print('=== K-Means cluster centroids ===')
display(km_profile)
print('\\n=== HDBSCAN cluster centroids ===')
display(hdb_profile)
"""),

code("""\
# ── radar chart helper ────────────────────────────────────────────────────────
def radar_chart(profile_df: pd.DataFrame, title: str, label_map: dict = {}):
    # normalise each column to [0,1] for radar
    norm = (profile_df - profile_df.min()) / (profile_df.max() - profile_df.min() + 1e-9)
    cols  = list(norm.columns)
    N     = len(cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    nrows = (len(norm) + 2) // 3
    fig   = plt.figure(figsize=(6 * min(3, len(norm)), 5 * nrows))

    for i, (cid, row) in enumerate(norm.iterrows()):
        ax = fig.add_subplot(nrows, min(3, len(norm)), i+1,
                             polar=True)
        vals = row.values.tolist() + [row.values[0]]
        color = '#aaaaaa' if cid == -1 else PALETTE[i % len(PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        short = [c.replace('_norm','').replace('_avg','').replace('_white','')
                   .replace('total_ply_count','game_length')[:14]
                 for c in cols]
        ax.set_xticklabels(short, fontsize=7)
        name  = label_map.get(int(cid), f'Cluster {cid}') if cid != -1 else 'Noise'
        ax.set_title(name, fontsize=9, pad=12)
        ax.set_yticks([])

    plt.suptitle(title, fontsize=13)
    plt.tight_layout(); plt.show()

if VERBOSE_PLOTS:
    radar_chart(km_profile,  f'K-Means Play-Style Radar (k={KMEANS_K})', KMEANS_LABELS)
    radar_chart(hdb_profile, 'HDBSCAN Play-Style Radar')
"""),

md("""## 11 · Archetype interpretation guide

After inspecting the centroids and radar charts, fill in `KMEANS_LABELS` in the CONFIG cell.

| Signature | Likely archetype |
|---|---|
| High `check_density_avg`, high `capture_aggression`, low `total_ply_count` | **Aggressive Attacker** |
| High `blunder_density` (engine on), high `time_pressure_avg` | **Blunder King / Time-Scrambler** |
| Low `avg_time_avg`, high `time_pressure_avg`, low `clock_remaining_avg` | **Fast & Loose** |
| Low `avg_time_avg`, low `time_pressure_avg`, high `avg_elo` | **Fast & Sharp** |
| High `avg_time_avg`, high `clock_remaining_avg`, long games | **Slow & Positional** |
| High `queen_moves_before_10`, low `castle_move_*` | **Gambit / Early Queen Rusher** |
| High `time_variability`, inconsistent `opening_pace` | **Erratic Thinker** |
| High `opening_pace_avg`, low `endgame_time_norm` | **Endgame Hoarder** |
"""),

md("## 12 · Save results"),

code("""\
if SAVE_CSV:
    # Attach t-SNE coords for the subsample; fill NaN for the rest
    df['tsne_x'] = np.nan
    df['tsne_y'] = np.nan
    df.loc[ts_idx if _subset else df.index, 'tsne_x'] = emb_tsne[:, 0]
    df.loc[ts_idx if _subset else df.index, 'tsne_y'] = emb_tsne[:, 1]

    save_cols = (
        ['WhiteElo', 'BlackElo', 'avg_elo']
        + _profile_cols
        + ['kmeans_cluster', 'hdbscan_cluster',
           'umap_x', 'umap_y', 'tsne_x', 'tsne_y']
    )
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_csv(OUTPUT_CSV, index=False)
    print(f'Saved → {OUTPUT_CSV}  ({len(df):,} rows)')
"""),
]

nb1 = new_notebook(cells=nb1_cells)
nb1.metadata['kernelspec'] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK 2 — chess_tier_classification.ipynb
# ═══════════════════════════════════════════════════════════════════════════════

nb2_cells = []

nb2_cells += [
md("""# Chess Player Tier Classification
Classifies each player's Elo into skill tiers using board features + clock features.

**Tiers**
| Label | Elo range |
|---|---|
| Beginner | < 1000 |
| Novice | 1000–1200 |
| Intermediate | 1200–1600 |
| Expert | 1600–1800 |
| Advanced | 1800–2000 |
| Pro | 2000–2200 |
| Master/GM | ≥ 2200 |

**Pipeline**
1. Same data loader + feature engineering as `chess_elo_v_final`
2. Tier label encoding
3. Train / Val / Test split + class weights
4. Pluggable model backend: **LightGBM** and **XGBoost** (toggle via `MODEL_BACKEND`)
5. Optional hyperparameter search
6. Confusion matrix, per-class metrics, feature importance
7. Auto-save trained model

> Toggle every behaviour from the **CONFIGURATION** cell — nowhere else.
"""),

code("""\
# pip install chess zstandard lightgbm xgboost pandas scikit-learn matplotlib seaborn joblib
import io, re, time, os, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import zstandard as zstd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint, uniform

import lightgbm as lgb
import xgboost as xgb
from joblib import Parallel, delayed

from chess_features_final import extract_features_dataframe, compute_elo_sample_weights

warnings.filterwarnings('ignore')
print('All imports OK')
"""),

md("## ⚙️ CONFIGURATION — edit here only"),

code("""\
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "../../data/villads_data/lichess_db_standard_rated_2017-11.pgn.zst"
MAX_GAMES    = 3_000_000
RANDOM_STATE = 42
N_JOBS       = -1

# ── splits ────────────────────────────────────────────────────────────────────
VAL_SIZE  = 0.10
TEST_SIZE = 0.10

# ── time-control filter ───────────────────────────────────────────────────────
FILTER_BASE_SECONDS  = 600
FILTER_SHORT_GAMES   = True
MIN_TOTAL_PLIES      = 12
EXCLUDE_TERMINATIONS = {"Time forfeit", "Abandoned", "Unterminated"}

# ── tier definition ───────────────────────────────────────────────────────────
# Defines ELO bin edges and human-readable tier names.
# Edit freely — the rest of the notebook adapts automatically.
TIER_BINS  = [0, 1000, 1200, 1600, 1800, 2000, 2200, 4000]
TIER_NAMES = ['Beginner', 'Novice', 'Intermediate', 'Expert', 'Advanced', 'Pro', 'Master/GM']

# Which Elo column to classify? 'WhiteElo' | 'BlackElo' | 'avg' (mean of both)
ELO_TARGET = 'avg'

# ── feature groups ────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    'structure':  True,
    'checks':     True,
    'captures':   True,
    'castling':   True,
    'style':      True,
    'clock':      True,
    'engine':     False,
    'meta':       False,
}
ADD_TIME_CONTROL_NUMERIC = True
ECO_COARSE_LEVEL         = "letter"  # only when meta=True

# ── model selection ───────────────────────────────────────────────────────────
# Switch between backends here — everything else is automatic.
MODEL_BACKEND = 'lgbm'   # 'lgbm'  or  'xgb'

# ── LightGBM params ───────────────────────────────────────────────────────────
LGBM_PARAMS = {
    'objective':         'multiclass',
    'num_class':         len(TIER_NAMES),
    'n_estimators':      600,
    'learning_rate':     0.05,
    'num_leaves':        63,
    'min_child_samples': 20,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'class_weight':      'balanced',
    'random_state':      RANDOM_STATE,
    'verbose':           -1,
    'n_jobs':            N_JOBS,
}

# ── XGBoost params ────────────────────────────────────────────────────────────
XGB_PARAMS = {
    'objective':        'multi:softprob',
    'num_class':        len(TIER_NAMES),
    'n_estimators':     600,
    'learning_rate':    0.05,
    'max_depth':        6,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'use_label_encoder': False,
    'eval_metric':      'mlogloss',
    'random_state':     RANDOM_STATE,
    'verbosity':        0,
    'n_jobs':           N_JOBS,
}

# ── hyperparameter search ─────────────────────────────────────────────────────
RUN_HYPERPARAM_SEARCH = False
HYPERPARAM_N_ITER     = 20
HYPERPARAM_CV_FOLDS   = 3

LGBM_SEARCH_SPACE = {
    'n_estimators':      randint(300, 900),
    'learning_rate':     uniform(0.02, 0.1),
    'num_leaves':        randint(31, 127),
    'min_child_samples': randint(10, 60),
    'subsample':         uniform(0.6, 0.4),
    'colsample_bytree':  uniform(0.6, 0.4),
}
XGB_SEARCH_SPACE = {
    'n_estimators':  randint(300, 900),
    'learning_rate': uniform(0.02, 0.1),
    'max_depth':     randint(4, 10),
    'subsample':     uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
}

# ── pretrained model ──────────────────────────────────────────────────────────
LOAD_PRETRAINED  = False
PRETRAINED_PATH  = None   # e.g. 'tier_clf_lgbm_20250101_120000.txt'

# ── feature importance ────────────────────────────────────────────────────────
RUN_PERM_IMPORTANCE        = True
FEATURE_IMPORTANCE_METHOD  = 'gain'   # 'gain' | 'split' | 'permutation'
PERM_N_REPEATS             = 5
TOP_N_FEATURES             = 20
RETRAIN_TOP_N              = False

# ── output ────────────────────────────────────────────────────────────────────
VERBOSE_PLOTS    = True
MODEL_SAVE_NAME  = f'tier_clf_{MODEL_BACKEND}'
MODEL_SAVE_NOTES = f'{MAX_GAMES // 1_000}k_600s'

# ═══ derived — do not edit ═════════════════════════════════════════════════════
_N_CLASSES = len(TIER_NAMES)
assert len(TIER_BINS) - 1 == _N_CLASSES, "TIER_BINS must have len(TIER_NAMES)+1 edges"
"""),

md("## 1 · Load data"),

code("""\
games_list = []
dctx = zstd.ZstdDecompressor()

t0 = time.time()
with open(DATA_PATH, 'rb') as compressed_file:
    with dctx.stream_reader(compressed_file) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        current_game = {}
        for line in text_stream:
            line = line.strip()
            if line.startswith('['):
                tag = line.split(' ')[0][1:]
                val = line.split('"')[1]
                if tag in ('WhiteElo', 'BlackElo', 'TimeControl', 'ECO',
                           'Termination', 'Result'):
                    current_game[tag] = val
            elif line.startswith('1.'):
                current_game['Moves'] = line
                if 'WhiteElo' in current_game and 'BlackElo' in current_game:
                    games_list.append(current_game)
                current_game = {}
                if len(games_list) >= MAX_GAMES:
                    break

df_raw = pd.DataFrame(games_list)
df_raw['WhiteElo'] = pd.to_numeric(df_raw['WhiteElo'], errors='coerce')
df_raw['BlackElo'] = pd.to_numeric(df_raw['BlackElo'], errors='coerce')
df_raw = df_raw.dropna(subset=['WhiteElo', 'BlackElo']).copy()
df_raw['WhiteElo'] = df_raw['WhiteElo'].astype(int)
df_raw['BlackElo'] = df_raw['BlackElo'].astype(int)
df_raw['Moves']    = df_raw['Moves'].fillna('').astype(str)
print(f'Loaded {len(df_raw):,} games in {time.time()-t0:.1f}s')

if FILTER_BASE_SECONDS is not None:
    _tc_base = (
        df_raw['TimeControl'].astype(str)
        .str.extract(r'^(\\d+)\\+')[0]
        .astype(float)
    )
    df_raw = df_raw[_tc_base == FILTER_BASE_SECONDS].copy().reset_index(drop=True)
    print(f'After TC filter ({FILTER_BASE_SECONDS}s): {len(df_raw):,} games')

if EXCLUDE_TERMINATIONS and 'Termination' in df_raw.columns:
    before = len(df_raw)
    df_raw = df_raw[~df_raw['Termination'].isin(EXCLUDE_TERMINATIONS)].copy().reset_index(drop=True)
    print(f'After termination filter: {len(df_raw):,} (dropped {before-len(df_raw):,})')

if ADD_TIME_CONTROL_NUMERIC and 'TimeControl' in df_raw.columns:
    _tc = df_raw['TimeControl'].astype(str).str.extract(r'^(\\d+)\\+(\\d+)')
    df_raw['tc_base']      = pd.to_numeric(_tc[0], errors='coerce').fillna(0).astype(float)
    df_raw['tc_increment'] = pd.to_numeric(_tc[1], errors='coerce').fillna(0).astype(float)

df_raw.head(3)
"""),

md("## 2 · Clock features"),

code("""\
_CLK_RE = re.compile(r'\\[%clk\\s+(\\d+):(\\d+):(\\d+)\\]')

def parse_clock_features(moves_string: str, time_control: str = '?') -> dict:
    clocks = [int(h)*3600 + int(m)*60 + int(s)
              for h, m, s in _CLK_RE.findall(moves_string)]
    clk_w, clk_b = clocks[0::2], clocks[1::2]
    tc_m      = re.match(r'(\\d+)\\+(\\d+)', str(time_control))
    base      = int(tc_m.group(1)) if tc_m else None
    increment = int(tc_m.group(2)) if tc_m else 0
    norm      = base if base else 1

    def _spent(seq, start):
        out, prev = [], start
        for c in seq:
            if prev is not None:
                s = prev - c + increment
                if s >= 0: out.append(s)
            prev = c
        return out

    sw = _spent(clk_w, base)
    sb = _spent(clk_b, base)

    def _stats(seq):
        if not seq: return 0., 0., 0.
        a = np.array(seq)
        return float(a.mean()), float(a.std()), float(a.max())

    def _trend(seq):
        if len(seq) < 2: return 0.0
        return float(np.polyfit(np.arange(len(seq)), np.array(seq, dtype=float), 1)[0])

    def _window_mean(seq, n=10, which='first'):
        if not seq: return 0.0
        return float(np.mean(seq[:n])) if which == 'first' else float(np.mean(seq[-n:]))

    aw, sw2, mw = _stats(sw); ab, sb2, mb = _stats(sb)

    return {
        'avg_time_norm_white':         aw / norm,
        'std_time_norm_white':         sw2 / norm,
        'max_time_norm_white':         mw / norm,
        'time_pressure_white':         sum(1 for c in clk_w if c < 10),
        'opening_pace_norm_white':     (np.mean(sw[:10]) if sw else 0.) / norm,
        'clock_remaining_norm_white':  (clk_w[-1] / norm) if clk_w else 0.,
        'opening_time_norm_white':     _window_mean(sw, 10, 'first') / norm,
        'endgame_time_norm_white':     _window_mean(sw, 10, 'last') / norm,
        'time_spent_trend_norm_white': _trend(sw) / norm,
        'avg_time_norm_black':         ab / norm,
        'std_time_norm_black':         sb2 / norm,
        'max_time_norm_black':         mb / norm,
        'time_pressure_black':         sum(1 for c in clk_b if c < 10),
        'opening_pace_norm_black':     (np.mean(sb[:10]) if sb else 0.) / norm,
        'clock_remaining_norm_black':  (clk_b[-1] / norm) if clk_b else 0.,
        'opening_time_norm_black':     _window_mean(sb, 10, 'first') / norm,
        'endgame_time_norm_black':     _window_mean(sb, 10, 'last') / norm,
        'time_spent_trend_norm_black': _trend(sb) / norm,
    }

if FEATURE_GROUPS['clock']:
    tc_col = df_raw['TimeControl'] if 'TimeControl' in df_raw.columns else ['?']*len(df_raw)
    clock_records = Parallel(n_jobs=N_JOBS)(
        delayed(parse_clock_features)(m, t)
        for m, t in zip(df_raw['Moves'], tc_col)
    )
    df_clocks = pd.DataFrame(clock_records, index=df_raw.index)
    print('Clock features:', df_clocks.shape)
else:
    df_clocks = pd.DataFrame(index=df_raw.index)
    print('Clock features skipped')
"""),

md("## 3 · Board features"),

code("""\
t0       = time.time()
df_feats = extract_features_dataframe(df_raw, n_jobs=N_JOBS)
elapsed  = time.time() - t0
print(f'Extracted {len(df_feats):,} games in {elapsed:.1f}s  ({elapsed/len(df_feats)*1000:.1f} ms/game)')
"""),

md("## 4 · Merge, label, build feature matrix"),

code("""\
_BOARD_FEATURES = {
    'structure': ['total_ply_count', 'material_balance_end', 'result_encoded'],
    'checks':    ['checks_given_white', 'checks_given_black',
                  'check_density_white', 'check_density_black'],
    'captures':  ['first_capture_move_white', 'first_capture_move_black',
                  'pawn_captures_total', 'piece_captures_total', 'capture_density'],
    'castling':  ['castle_move_white', 'castle_move_black'],
    'style':     ['consec_same_piece_white', 'consec_same_piece_black',
                  'queen_moves_before_10', 'white_territory_depth', 'black_territory_depth',
                  'promotions', 'en_passant_captures',
                  'legal_moves_white_move5', 'legal_moves_black_move5'],
    'engine':    ['acpl_white', 'inaccuracy_count_white', 'mistake_count_white',
                  'blunder_count_white', 'blunder_density_white',
                  'acpl_black', 'inaccuracy_count_black', 'mistake_count_black',
                  'blunder_count_black', 'blunder_density_black'],
}
_CLOCK_FEATURES = [
    'avg_time_norm_white', 'std_time_norm_white', 'max_time_norm_white',
    'time_pressure_white', 'opening_pace_norm_white', 'clock_remaining_norm_white',
    'opening_time_norm_white', 'endgame_time_norm_white', 'time_spent_trend_norm_white',
    'avg_time_norm_black', 'std_time_norm_black', 'max_time_norm_black',
    'time_pressure_black', 'opening_pace_norm_black', 'clock_remaining_norm_black',
    'opening_time_norm_black', 'endgame_time_norm_black', 'time_spent_trend_norm_black',
]
_CASTLE_CAT = ['castle_side_white', 'castle_side_black'] if FEATURE_GROUPS['castling'] else []
_META_CAT   = []
if FEATURE_GROUPS['meta']:
    _META_CAT = ['TimeControl', 'ECO_Group'] if ECO_COARSE_LEVEL == 'letter' else ['TimeControl', 'ECO']
_TC_NUMERIC = ['tc_base', 'tc_increment'] if ADD_TIME_CONTROL_NUMERIC else []

NUMERIC_FEATURES     = []
for grp, cols in _BOARD_FEATURES.items():
    if FEATURE_GROUPS.get(grp, False):
        NUMERIC_FEATURES += cols
if FEATURE_GROUPS['clock']:
    NUMERIC_FEATURES += _CLOCK_FEATURES
NUMERIC_FEATURES += _TC_NUMERIC
CATEGORICAL_FEATURES = _CASTLE_CAT + _META_CAT

# ── merge ──────────────────────────────────────────────────────────────────────
meta_cols = ['WhiteElo', 'BlackElo']
if FEATURE_GROUPS['meta']:
    meta_cols += ['TimeControl', 'ECO']
if ADD_TIME_CONTROL_NUMERIC:
    meta_cols += ['tc_base', 'tc_increment']
meta_cols = list(dict.fromkeys([c for c in meta_cols if c in df_raw.columns]))

df = (
    df_raw[meta_cols]
    .join(df_feats, how='inner')
    .join(df_clocks, how='inner')
)
df = df.replace([np.inf, -np.inf], np.nan)

if FEATURE_GROUPS['meta'] and ECO_COARSE_LEVEL == 'letter' and 'ECO' in df.columns:
    df['ECO_Group'] = df['ECO'].astype(str).str[0].fillna('U')

if FILTER_SHORT_GAMES and 'total_ply_count' in df.columns:
    before = len(df)
    df = df[df['total_ply_count'] >= MIN_TOTAL_PLIES].copy().reset_index(drop=True)
    print(f'Short-game filter (<{MIN_TOTAL_PLIES} plies): dropped {before-len(df):,}')

# ── Elo target → tier label ────────────────────────────────────────────────────
if ELO_TARGET == 'avg':
    elo_vals = (df['WhiteElo'] + df['BlackElo']) / 2
elif ELO_TARGET == 'WhiteElo':
    elo_vals = df['WhiteElo']
else:
    elo_vals = df['BlackElo']

df['elo_for_tier'] = elo_vals
df['tier_label']   = pd.cut(
    elo_vals,
    bins=TIER_BINS,
    labels=TIER_NAMES,
    right=True,
)
df = df.dropna(subset=['tier_label']).copy().reset_index(drop=True)
df['tier_int'] = df['tier_label'].cat.codes   # integer label for models

print(f'DataFrame: {df.shape}')
print('Tier distribution:')
print(df['tier_label'].value_counts().sort_index())

if VERBOSE_PLOTS:
    fig, ax = plt.subplots(figsize=(10, 4))
    df['tier_label'].value_counts().loc[TIER_NAMES].plot(
        kind='bar', ax=ax, color='#4e91d9', edgecolor='none', alpha=0.85
    )
    ax.set_xlabel('Tier'); ax.set_ylabel('Games')
    ax.set_title('Tier distribution')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(); plt.show()
"""),

md("## 5 · Preprocessor + feature matrix"),

code("""\
def build_preprocessor(numeric_cols, categorical_cols):
    transformers = [('num', SimpleImputer(strategy='median'), numeric_cols)]
    if categorical_cols:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        )
    return ColumnTransformer(transformers=transformers, remainder='drop')

_num_cols = [c for c in NUMERIC_FEATURES    if c in df.columns]
_cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
preprocessor = build_preprocessor(_num_cols, _cat_cols)
X_all        = preprocessor.fit_transform(df)
feat_names   = list(preprocessor.get_feature_names_out())

y_all = df['tier_int'].values
print(f'Feature matrix : {X_all.shape}')
print(f'Classes (0-idx): {np.unique(y_all)}')
"""),

md("## 6 · Train / Val / Test split + sample weights"),

code("""\
idx = np.arange(len(df))

idx_tv, idx_test = train_test_split(
    idx, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
)
_val_frac = VAL_SIZE / (1 - TEST_SIZE)
idx_train, idx_val = train_test_split(
    idx_tv, test_size=_val_frac, stratify=y_all[idx_tv], random_state=RANDOM_STATE
)

X_train, X_val, X_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]
y_train, y_val, y_test  = y_all[idx_train], y_all[idx_val], y_all[idx_test]

# Balanced class weights for training
sample_weights = compute_sample_weight('balanced', y_train)

print(f'Train: {len(idx_train):,}  Val: {len(idx_val):,}  Test: {len(idx_test):,}')
print('Tier counts (test set):', dict(zip(TIER_NAMES, np.bincount(y_test, minlength=_N_CLASSES))))

if VERBOSE_PLOTS:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, split, y_split in zip(axes, ['Train', 'Val', 'Test'],
                                   [y_train, y_val, y_test]):
        counts = np.bincount(y_split, minlength=_N_CLASSES)
        ax.bar(TIER_NAMES, counts, color='#4e91d9', alpha=0.8)
        ax.set_title(f'{split} split'); ax.set_xlabel('Tier'); ax.set_ylabel('count')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    plt.tight_layout(); plt.show()
"""),

md("## 7 · Hyperparameter search (toggle `RUN_HYPERPARAM_SEARCH`)"),

code("""\
if MODEL_BACKEND == 'lgbm':
    _base_clf   = lgb.LGBMClassifier(**{k:v for k,v in LGBM_PARAMS.items()
                                         if k not in ('num_class',)})
    _search_spc = LGBM_SEARCH_SPACE
else:
    _base_clf   = xgb.XGBClassifier(**{k:v for k,v in XGB_PARAMS.items()
                                        if k not in ('num_class',)})
    _search_spc = XGB_SEARCH_SPACE

best_params = (LGBM_PARAMS if MODEL_BACKEND == 'lgbm' else XGB_PARAMS).copy()

if RUN_HYPERPARAM_SEARCH and not LOAD_PRETRAINED:
    print(f'Randomised search ({MODEL_BACKEND.upper()}) — {HYPERPARAM_N_ITER} iters...')
    search = RandomizedSearchCV(
        _base_clf, _search_spc,
        n_iter=HYPERPARAM_N_ITER,
        cv=StratifiedKFold(n_splits=HYPERPARAM_CV_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE),
        scoring='balanced_accuracy',
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train, sample_weight=sample_weights)
    best_params.update(search.best_params_)
    print('Best params:', search.best_params_)
    print(f'Best CV balanced-acc: {search.best_score_:.4f}')
elif LOAD_PRETRAINED:
    print('Hyperparam search skipped — loading pretrained model')
else:
    print(f'Hyperparam search skipped — using {MODEL_BACKEND.upper()} default params')
"""),

md("## 8 · Train"),

code("""\
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── model factory ──────────────────────────────────────────────────────────────
def make_model(backend: str, params: dict):
    \"\"\"Return a fresh (unfitted) classifier for the given backend.\"\"\
"
    if backend == 'lgbm':
        kw = {k: v for k, v in params.items() if k != 'num_class'}
        return lgb.LGBMClassifier(**kw)
    elif backend == 'xgb':
        kw = {k: v for k, v in params.items() if k != 'num_class'}
        return xgb.XGBClassifier(**kw)
    raise ValueError(f'Unknown backend: {backend}')


def fit_model(model, backend: str):
    \"\"\"Backend-aware fitting with early stopping.\"\"\
"
    if backend == 'lgbm':
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
    elif backend == 'xgb':
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    return model


def save_model(model, backend: str, suffix: str = '') -> str:
    fname = f'{MODEL_SAVE_NAME}_{MODEL_SAVE_NOTES}{suffix}_{_ts}'
    if backend == 'lgbm':
        fname += '.txt'
        model.booster_.save_model(fname)
    elif backend == 'xgb':
        fname += '.ubj'
        model.save_model(fname)
    return fname


def load_model(path: str, backend: str, params: dict):
    model = make_model(backend, params)
    if backend == 'lgbm':
        model.booster_ = lgb.Booster(model_file=path)
        n = model.booster_.num_feature()
        model._n_features = n; model._n_features_in = n
    elif backend == 'xgb':
        model.load_model(path)
    return model


# ── train or load ──────────────────────────────────────────────────────────────
if LOAD_PRETRAINED:
    if not PRETRAINED_PATH or not os.path.exists(PRETRAINED_PATH):
        raise FileNotFoundError(f'Model not found: {PRETRAINED_PATH}')
    print(f'Loading pretrained {MODEL_BACKEND.upper()} from {PRETRAINED_PATH}...')
    clf = load_model(PRETRAINED_PATH, MODEL_BACKEND, best_params)
else:
    print(f'Training {MODEL_BACKEND.upper()}...')
    t0  = time.time()
    clf = make_model(MODEL_BACKEND, best_params)
    clf = fit_model(clf, MODEL_BACKEND)
    elapsed = time.time() - t0
    print(f'Training done in {elapsed:.1f}s')
    fname = save_model(clf, MODEL_BACKEND)
    print(f'Saved → {fname}')
"""),

md("## 9 · Evaluate"),

code("""\
val_proba  = clf.predict_proba(X_val)
test_proba = clf.predict_proba(X_test)

val_preds  = np.argmax(val_proba,  axis=1)
test_preds = np.argmax(test_proba, axis=1)

val_acc    = accuracy_score(y_val,  val_preds)
test_acc   = accuracy_score(y_test, test_preds)
test_bal   = balanced_accuracy_score(y_test, test_preds)
test_top2  = top_k_accuracy_score(y_test, test_proba, k=2)
test_top3  = top_k_accuracy_score(y_test, test_proba, k=3)

print(f'  Val  Accuracy          : {val_acc:.4f}')
print(f'  Test Accuracy          : {test_acc:.4f}')
print(f'  Test Balanced-Accuracy : {test_bal:.4f}')
print(f'  Test Top-2 Accuracy    : {test_top2:.4f}')
print(f'  Test Top-3 Accuracy    : {test_top3:.4f}')
print()
print(classification_report(y_test, test_preds,
                             target_names=TIER_NAMES,
                             digits=3))
"""),

code("""\
if VERBOSE_PLOTS:
    # ── confusion matrix ───────────────────────────────────────────────────────
    cm      = confusion_matrix(y_test, test_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mat, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        ['Confusion Matrix — counts', 'Confusion Matrix — row-normalised'],
    ):
        sns.heatmap(
            mat, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=TIER_NAMES, yticklabels=TIER_NAMES,
            ax=ax, cbar=True,
        )
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout(); plt.show()

    # ── per-class probability distributions ───────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()
    for i, name in enumerate(TIER_NAMES):
        ax    = axes_flat[i]
        mask  = y_test == i
        ax.hist(test_proba[mask,  i], bins=40, alpha=0.7, color='#4e91d9',
                label='Correct class')
        ax.hist(test_proba[~mask, i], bins=40, alpha=0.5, color='#e05252',
                label='Other classes')
        ax.set_title(f'{name} — P(class)')
        ax.set_xlabel('Predicted probability'); ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(TIER_NAMES), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle('Predicted Probability Distributions per Class', fontsize=13)
    plt.tight_layout(); plt.show()
"""),

md("## 10 · Feature importance"),

code("""\
perm_df = None

if RUN_PERM_IMPORTANCE:
    _method = str(FEATURE_IMPORTANCE_METHOD).strip().lower()

    def compute_importance(method: str) -> pd.DataFrame:
        if method == 'permutation':
            print(f'Permutation importance ({PERM_N_REPEATS} repeats)...')
            t0 = time.time()
            result = permutation_importance(
                clf, X_test, y_test,
                n_repeats=PERM_N_REPEATS,
                scoring='balanced_accuracy',
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            )
            print(f'  done in {time.time()-t0:.1f}s')
            n = len(feat_names)
            return pd.DataFrame({
                'feature':    feat_names[:n],
                'importance': result.importances_mean[:n],
                'std':        result.importances_std[:n],
            }).sort_values('importance', ascending=False).reset_index(drop=True)

        if MODEL_BACKEND == 'lgbm':
            booster    = clf.booster_
            importances = booster.feature_importance(importance_type=method)
        elif MODEL_BACKEND == 'xgb':
            itype = 'total_gain' if method == 'gain' else 'weight'
            scores = clf.get_booster().get_score(importance_type=itype)
            importances = np.array([scores.get(f, 0.) for f in feat_names])

        n = len(feat_names)
        return pd.DataFrame({
            'feature':    feat_names[:n],
            'importance': importances[:n],
            'std':        np.zeros(n, dtype=float),
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    perm_df = compute_importance(_method)

    if VERBOSE_PLOTS:
        top = perm_df.head(TOP_N_FEATURES).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top['feature'], top['importance'], xerr=top['std'],
                color='#4e91d9', ecolor='#333', capsize=3, alpha=0.85)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(f'Top {TOP_N_FEATURES} features — {_method} ({MODEL_BACKEND.upper()})')
        ax.set_xlabel('Importance')
        plt.tight_layout(); plt.show()
else:
    print('Feature importance skipped (RUN_PERM_IMPORTANCE=False)')
"""),

md("## 11 · Retrain on top-N features (toggle `RETRAIN_TOP_N`)"),

code("""\
if RETRAIN_TOP_N and perm_df is not None:
    top_feats = perm_df.head(TOP_N_FEATURES)['feature'].tolist()
    print(f'Retraining on top {TOP_N_FEATURES} features: {top_feats}')

    _top_idx  = [feat_names.index(f) for f in top_feats if f in feat_names]
    X_tr_top  = X_all[idx_train][:, _top_idx]
    X_v_top   = X_all[idx_val][:,   _top_idx]
    X_te_top  = X_all[idx_test][:,  _top_idx]

    clf_top = make_model(MODEL_BACKEND, best_params)
    if MODEL_BACKEND == 'lgbm':
        clf_top.fit(
            X_tr_top, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_v_top, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
    else:
        clf_top.fit(
            X_tr_top, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_v_top, y_val)],
            verbose=False,
        )

    top_preds = np.argmax(clf_top.predict_proba(X_te_top), axis=1)
    print(f'Top-{TOP_N_FEATURES} test accuracy : {accuracy_score(y_test, top_preds):.4f}')
    print(f'Top-{TOP_N_FEATURES} balanced-acc  : {balanced_accuracy_score(y_test, top_preds):.4f}')
    fname = save_model(clf_top, MODEL_BACKEND, suffix=f'_top{TOP_N_FEATURES}')
    print(f'Saved → {fname}')
else:
    if not RETRAIN_TOP_N:
        print('Top-N retrain skipped (RETRAIN_TOP_N=False)')
    else:
        print('Top-N retrain skipped — run feature importance first')
"""),

md("## 12 · Summary table"),

code("""\
rows = [
    {
        'run':             f'{MODEL_BACKEND.upper()} full',
        'features':        X_all.shape[1],
        'val_accuracy':    val_acc,
        'test_accuracy':   test_acc,
        'balanced_acc':    test_bal,
        'top2_accuracy':   test_top2,
        'top3_accuracy':   test_top3,
    }
]
if RETRAIN_TOP_N and perm_df is not None:
    rows.append({
        'run':           f'{MODEL_BACKEND.upper()} top-{TOP_N_FEATURES}',
        'features':      TOP_N_FEATURES,
        'test_accuracy': accuracy_score(y_test, top_preds),
        'balanced_acc':  balanced_accuracy_score(y_test, top_preds),
    })
pd.DataFrame(rows).round(4)
"""),
]

nb2 = new_notebook(cells=nb2_cells)
nb2.metadata['kernelspec'] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}

# ── write files ────────────────────────────────────────────────────────────────
OUT1 = 'C:\\Users\\villa\\Desktop\\AppML26\\AppML_FinalProject\\root\\workspace\\villads_sandbox\\out_files\\chess_playstyle_clustering.ipynb'
OUT2 = 'C:\\Users\\villa\\Desktop\\AppML26\\AppML_FinalProject\\root\\workspace\\villads_sandbox\\out_files\\chess_tier_classification.ipynb'

os.makedirs(os.path.dirname(OUT1), exist_ok=True)
    
with open(OUT1, 'w', encoding='utf-8') as f:
    nbformat.write(nb1, f)

with open(OUT2, 'w', encoding='utf-8') as f:
    nbformat.write(nb2, f)

print("Done:", OUT1)
print("Done:", OUT2)
