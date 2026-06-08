from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_feature_matrix(df: pd.DataFrame, columns: Sequence[str], *, scale: bool = True):
    X = df[list(columns)].replace([np.inf, -np.inf], np.nan)
    steps = [SimpleImputer(strategy="median")]
    if scale:
        steps.append(StandardScaler())
    pipe = make_pipeline(*steps)
    return pipe.fit_transform(X), pipe


def compute_pca_embeddings(X: np.ndarray, *, random_state: int = 42) -> dict[str, pd.DataFrame]:
    out = {}
    for n in [2, 3]:
        pca = PCA(n_components=n, random_state=random_state)
        coords = pca.fit_transform(X)
        out[f"pca{n}"] = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(n)])
    return out


def compute_umap_embeddings(
    X: np.ndarray,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    metric: str = "euclidean",
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install UMAP with: pip install umap-learn") from exc

    out = {}
    for n in [2, 3]:
        reducer = umap.UMAP(n_components=n, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
        coords = reducer.fit_transform(X)
        out[f"umap{n}"] = pd.DataFrame(coords, columns=[f"UMAP{i+1}" for i in range(n)])
    return out


def compute_embeddings(X: np.ndarray, *, include_umap: bool = True, random_state: int = 42) -> dict[str, pd.DataFrame]:
    out = compute_pca_embeddings(X, random_state=random_state)
    if include_umap:
        out.update(compute_umap_embeddings(X, random_state=random_state))
    return out


def attach_metadata(embedding: pd.DataFrame, metadata: pd.DataFrame, columns: Sequence[str] = ("player", "elo_mean", "elo_median", "elo_std")) -> pd.DataFrame:
    out = embedding.copy()
    for col in columns:
        if col in metadata:
            out[col] = metadata[col].to_numpy()
    return out


def plot_embedding_2d(df: pd.DataFrame, x: str, y: str, *, color_by: str | None = None, title: str | None = None, s: float = 8, alpha: float = 0.75):
    fig, ax = plt.subplots(figsize=(7, 5))
    if color_by and color_by in df:
        sc = ax.scatter(df[x], df[y], c=df[color_by], s=s, alpha=alpha)
        fig.colorbar(sc, ax=ax, label=color_by)
    else:
        ax.scatter(df[x], df[y], s=s, alpha=alpha)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{x} vs {y}")
    fig.tight_layout()
    return fig, ax


def plot_embedding_3d(df: pd.DataFrame, x: str, y: str, z: str, *, color_by: str | None = None, title: str | None = None, s: float = 8, alpha: float = 0.75):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    if color_by and color_by in df:
        sc = ax.scatter(df[x], df[y], df[z], c=df[color_by], s=s, alpha=alpha)
        fig.colorbar(sc, ax=ax, label=color_by)
    else:
        ax.scatter(df[x], df[y], df[z], s=s, alpha=alpha)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(title or f"{x} / {y} / {z}")
    fig.tight_layout()
    return fig, ax

# ---------------------------------------------------------------------------
# Interactive plotting helpers.
# ---------------------------------------------------------------------------

def _plotly_hover_columns(df: pd.DataFrame, hover_cols: Sequence[str] | None) -> list[str]:
    if hover_cols is None:
        hover_cols = (
            "player", "elo_mean", "elo_median", "elo_std", "n_games", "win_rate",
            "cluster",
        )
    return [col for col in hover_cols if col in df.columns]


def _maybe_downsample_for_plotly(
    df: pd.DataFrame,
    *,
    max_points: int | None = 50_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep interactive plots responsive for very large player tables."""
    if max_points is None or len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=random_state).reset_index(drop=True)


def plot_embedding_2d_interactive(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color_by: str | None = None,
    title: str | None = None,
    hover_cols: Sequence[str] | None = None,
    s: float = 6,
    alpha: float = 0.75,
    max_points: int | None = 50_000,
    random_state: int = 42,
    show: bool = True,
):
    """
    Interactive 2D embedding plot using Plotly.

    Use this in the notebook when you want hover labels, zoom/pan, box select,
    and browser-exportable HTML plots. For very large player tables, the plot is
    downsampled by default to keep it responsive.
    """
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("Interactive plots require plotly. Install with: pip install plotly") from exc

    plot_df = _maybe_downsample_for_plotly(df, max_points=max_points, random_state=random_state).copy()
    color = color_by if color_by and color_by in plot_df.columns else None
    hover_data = _plotly_hover_columns(plot_df, hover_cols)

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color,
        hover_data=hover_data,
        title=title or f"{x} vs {y}",
        render_mode="webgl",
    )
    fig.update_traces(marker={"size": s, "opacity": alpha})
    fig.update_layout(template="plotly_white", width=850, height=620)
    if show:
        fig.show()
    return fig


def plot_embedding_3d_interactive(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    *,
    color_by: str | None = None,
    title: str | None = None,
    hover_cols: Sequence[str] | None = None,
    s: float = 3,
    alpha: float = 0.75,
    max_points: int | None = 30_000,
    random_state: int = 42,
    show: bool = True,
):
    """
    Interactive 3D embedding plot using Plotly.

    3D browser plots become heavy faster than 2D plots, so this function uses a
    lower default max_points than the 2D version.
    """
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("Interactive plots require plotly. Install with: pip install plotly") from exc

    plot_df = _maybe_downsample_for_plotly(df, max_points=max_points, random_state=random_state).copy()
    color = color_by if color_by and color_by in plot_df.columns else None
    hover_data = _plotly_hover_columns(plot_df, hover_cols)

    fig = px.scatter_3d(
        plot_df,
        x=x,
        y=y,
        z=z,
        color=color,
        hover_data=hover_data,
        title=title or f"{x} / {y} / {z}",
    )
    fig.update_traces(marker={"size": s, "opacity": alpha})
    fig.update_layout(template="plotly_white", width=900, height=700)
    if show:
        fig.show()
    return fig
