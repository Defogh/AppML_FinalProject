from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def evaluate_against_elo(player_df: pd.DataFrame, labels: Sequence[int], *, elo_col: str = "elo_mean") -> pd.DataFrame:
    """Use Elo only after clustering, as an external interpretability check."""
    df = player_df[[elo_col]].copy()
    df["cluster"] = labels
    return df.groupby("cluster").agg(
        n=(elo_col, "count"),
        elo_mean=(elo_col, "mean"),
        elo_median=(elo_col, "median"),
        elo_std=(elo_col, "std"),
        elo_min=(elo_col, "min"),
        elo_max=(elo_col, "max"),
    ).reset_index()


def cluster_feature_summary(player_df: pd.DataFrame, labels: Sequence[int], feature_cols: Sequence[str]) -> pd.DataFrame:
    df = player_df[list(feature_cols)].copy()
    df["cluster"] = labels
    return df.groupby("cluster").mean(numeric_only=True).reset_index()


def recommended_evaluation_steps() -> list[str]:
    return [
        "Do not use Elo, result, title, or opponent metadata as clustering inputs.",
        "Report internal metrics: silhouette, Davies-Bouldin, Calinski-Harabasz, HDBSCAN noise fraction and cluster-size balance.",
        "Run stability tests: recluster bootstrap/resampled players and compare labels with adjusted Rand index.",
        "Check interpretability: inspect feature centroids for each cluster and verify that the differences are chess-readable.",
        "Use Elo only after clustering: compare Elo distributions per cluster to test whether playstyle correlates with strength.",
        "Validate across time controls/months: clusters found in 10+0 should reappear when using a second 10+0 sample.",
        "Inspect examples: for each cluster, print representative players/games nearest to the cluster centroid.",
    ]
