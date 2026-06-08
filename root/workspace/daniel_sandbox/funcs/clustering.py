from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score


def run_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 30,
    min_samples: int | None = None,
    metric: str = "euclidean",
):
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("Install HDBSCAN with: pip install hdbscan") from exc
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, prediction_data=True)
    labels = model.fit_predict(X)
    return labels, model


def run_kmeans(X: np.ndarray, *, n_clusters: int, random_state: int = 42, n_init: str | int = "auto"):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    return labels, model


def cluster_size_table(labels: Sequence[int]) -> pd.DataFrame:
    s = pd.Series(labels, name="cluster")
    out = s.value_counts().sort_index().rename_axis("cluster").reset_index(name="n")
    out["frac"] = out["n"] / len(s) if len(s) else 0.0
    return out


def clustering_scores(X: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    labels = np.asarray(labels)
    mask = labels != -1
    usable = labels[mask]
    X_use = X[mask]
    n_clusters = len(set(usable))
    out = {
        "n_points": float(len(labels)),
        "n_clustered": float(mask.sum()),
        "noise_fraction": float(np.mean(labels == -1)) if len(labels) else np.nan,
        "n_clusters": float(n_clusters),
    }
    if n_clusters >= 2 and len(usable) > n_clusters:
        out["silhouette"] = float(silhouette_score(X_use, usable))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X_use, usable))
        out["davies_bouldin"] = float(davies_bouldin_score(X_use, usable))
    return out


def print_cluster_report(X: np.ndarray, labels: Sequence[int], *, name: str = "clustering") -> pd.DataFrame:
    scores = clustering_scores(X, labels)
    print(f"{name}")
    for key, value in scores.items():
        print(f"  {key}: {value:.4g}" if isinstance(value, float) else f"  {key}: {value}")
    sizes = cluster_size_table(labels)
    print("\nCluster sizes:")
    print(sizes.to_string(index=False))
    return sizes


def kmeans_sweep(X: np.ndarray, k_values: Sequence[int] = range(2, 16), *, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_values:
        labels, model = run_kmeans(X, n_clusters=int(k), random_state=random_state)
        scores = clustering_scores(X, labels)
        rows.append({"k": int(k), "inertia": float(model.inertia_), **scores})
    return pd.DataFrame(rows)


def plot_kmeans_sweep(sweep_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep_df["k"], sweep_df["inertia"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    ax.set_title("KMeans elbow plot")
    fig.tight_layout()
    return fig, ax


def plot_clustered_embedding(
    embedding_df: pd.DataFrame,
    labels: Sequence[int],
    *,
    dims: Sequence[str] | None = None,
    title: str = "Clustered embedding",
    s: float = 8,
    alpha: float = 0.75,
):
    labels = np.asarray(labels)
    if dims is None:
        numeric = embedding_df.select_dtypes(include=[np.number]).columns.tolist()
        dims = numeric[:2]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(embedding_df[dims[0]], embedding_df[dims[1]], c=labels, s=s, alpha=alpha)
    fig.colorbar(sc, ax=ax, label="cluster")
    ax.set_xlabel(dims[0])
    ax.set_ylabel(dims[1])
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def feature_centroids(player_df: pd.DataFrame, labels: Sequence[int], feature_cols: Sequence[str], *, top_n: int = 8) -> pd.DataFrame:
    df = player_df[list(feature_cols)].copy()
    df["cluster"] = labels
    global_mean = df.loc[df["cluster"] != -1, list(feature_cols)].mean()
    global_std = df.loc[df["cluster"] != -1, list(feature_cols)].std().replace(0, np.nan)
    rows = []
    for cluster, group in df.groupby("cluster"):
        if cluster == -1:
            continue
        z = ((group[list(feature_cols)].mean() - global_mean) / global_std).sort_values(key=lambda s: s.abs(), ascending=False)
        for feature, value in z.head(top_n).items():
            rows.append({"cluster": cluster, "feature": feature, "z_vs_global": value, "cluster_mean": group[feature].mean(), "global_mean": global_mean[feature]})
    return pd.DataFrame(rows)


def stability_by_resampling(
    X: np.ndarray,
    cluster_fn,
    *,
    n_runs: int = 10,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simple clustering stability check using adjusted Rand index on overlapping resamples."""
    rng = np.random.default_rng(random_state)
    runs = []
    n = X.shape[0]
    for i in range(n_runs):
        idx = np.sort(rng.choice(n, size=max(2, int(sample_frac * n)), replace=False))
        labels = np.asarray(cluster_fn(X[idx]))
        runs.append((idx, labels))

    rows = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            idx_i, lab_i = runs[i]
            idx_j, lab_j = runs[j]
            common, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(common) > 1:
                rows.append({"run_i": i, "run_j": j, "overlap": len(common), "ari": adjusted_rand_score(lab_i[pos_i], lab_j[pos_j])})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Interactive cluster plotting.
# ---------------------------------------------------------------------------

def plot_clustered_embedding_interactive(
    embedding_df: pd.DataFrame,
    labels: Sequence[int],
    *,
    dims: Sequence[str] | None = None,
    title: str = "Clustered embedding",
    hover_cols: Sequence[str] | None = None,
    s: float = 6,
    alpha: float = 0.75,
    max_points: int | None = 50_000,
    random_state: int = 42,
    show: bool = True,
):
    """Interactive 2D cluster plot using Plotly."""
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("Interactive plots require plotly. Install with: pip install plotly") from exc

    plot_df = embedding_df.copy()
    plot_df["cluster"] = pd.Series(labels, index=plot_df.index).astype(str)

    if dims is None:
        numeric = plot_df.select_dtypes(include=[np.number]).columns.tolist()
        dims = numeric[:2]

    if max_points is not None and len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=random_state).reset_index(drop=True)

    if hover_cols is None:
        hover_cols = ("player", "elo_mean", "elo_median", "elo_std", "n_games", "win_rate")
    hover_data = [col for col in hover_cols if col in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x=dims[0],
        y=dims[1],
        color="cluster",
        hover_data=hover_data,
        title=title,
        render_mode="webgl",
    )
    fig.update_traces(marker={"size": s, "opacity": alpha})
    fig.update_layout(template="plotly_white", width=850, height=620)
    if show:
        fig.show()
    return fig
