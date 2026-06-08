"""
Clustering table builders for the project workflow.

Public outputs:
    parse_to_game_data(...)     -> one row per game
    parse_to_player_data(...)   -> one row per player

There is intentionally no public player-game table. A two-row-per-game frame is
created internally only when aggregating games to players.

This file wraps clustering_updated.py and adds optional board-aware features from
chess_board_features_workflow.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

import clustering_updated as _base

PLAYER_SUMMARY_FEATURES_FAST = list(_base.PLAYER_SUMMARY_FEATURES)

PLAYER_SUMMARY_FEATURES_BOARD_AWARE = [
    "n_games",
    "median_elo",
    "elo_std",
    "mean_result",
    "mean_game_length",
    "mean_initial_seconds",
    "mean_increment_seconds",
    "capture_rate",
    "check_rate",
    "castle_rate",
    "first_move_diversity_rate",
    "eco_entropy",
    "speed_entropy",
    "mean_own_acpl",
    "mean_own_blunder_density",
    "mean_own_territory_depth",
    "mean_own_legal_moves_move5",
    "mean_own_consec_same_piece",
    "mean_own_first_capture_move",
    "mean_own_material_balance_end",
]


def parse_to_game_data(
    raw_df: pd.DataFrame,
    *,
    include_text_columns: bool = True,
    include_board_features: bool = False,
    board_moves_col: str | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Build one row per game.

    By default, this uses the fast SAN/comment-derived feature path. Set
    ``include_board_features=True`` to append slower board-aware features from
    python-chess. That mode is best used on the 300K training sample, especially
    if the sample CSV contains ``moves_pgn``.
    """
    game_df = _base.parse_to_game_data(raw_df, include_text_columns=include_text_columns)

    if not include_board_features:
        return game_df

    from chess_board_features_workflow import extract_board_features_dataframe

    board_df = extract_board_features_dataframe(
        raw_df.reset_index(drop=True),
        moves_col=board_moves_col,
        result_col=None,
        n_jobs=n_jobs,
    ).reset_index(drop=True)

    # Keep friend-code feature names where possible. If a future base feature
    # has the same name, suffix the board-aware version to avoid silent overwrite.
    overlap = [column for column in board_df.columns if column in game_df.columns]
    if overlap:
        board_df = board_df.rename(columns={column: f"board_{column}" for column in overlap})

    return pd.concat([game_df.reset_index(drop=True), board_df], axis=1)


def _make_player_game_frame(game_df: pd.DataFrame) -> pd.DataFrame:
    """Private helper: two rows per game, one per player perspective."""
    common_cols = [
        "game_id", "date", "time", "speed", "initial_seconds", "increment_seconds",
        "termination", "eco", "eco_family", "opening", "opening_san_4ply",
        "num_halfmoves", "eval_final_pawns", "eval_mean_abs_pawns", "eval_swing_pawns",
        "balanced_eval_frac", "queen_moves_before_10", "promotions", "en_passant_captures",
    ]
    common = {column: game_df[column] for column in common_cols if column in game_df}

    player_rows = []
    for own, opp, sign in [("white", "black", 1.0), ("black", "white", -1.0)]:
        row = pd.DataFrame({
            **common,
            "player": game_df[own],
            "opponent": game_df[opp],
            "color": own,
            "is_white": int(own == "white"),
            "elo": game_df[f"{own}_elo"],
            "opponent_elo": game_df[f"{opp}_elo"],
            "elo_diff_vs_opponent": game_df[f"{own}_elo"] - game_df[f"{opp}_elo"],
            "result_score": game_df["result_white_score"] if own == "white" else 1.0 - game_df["result_white_score"],
            "rating_diff": game_df[f"{own}_rating_diff"],
            "own_moves": game_df[f"num_{own}_moves"],
            "opp_moves": game_df[f"num_{opp}_moves"],
            "own_captures": game_df[f"{own}_captures"],
            "opp_captures": game_df[f"{opp}_captures"],
            "own_checks": game_df[f"{own}_checks"],
            "opp_checks": game_df[f"{opp}_checks"],
            "own_checkmates": game_df[f"{own}_checkmates"],
            "opp_checkmates": game_df[f"{opp}_checkmates"],
            "own_castled": game_df[f"{own}_castled"],
            "opp_castled": game_df[f"{opp}_castled"],
            "own_castle_side": game_df[f"{own}_castle_side"],
            "opp_castle_side": game_df[f"{opp}_castle_side"],
            "first_own_move": game_df[f"first_{own}_move"],
            "first_opp_move": game_df[f"first_{opp}_move"],
        })

        for suffix in [
            "clock_last_seconds", "clock_min_seconds", "clock_mean_seconds",
            "clock_drop_observed_seconds", "clock_frac_under_10s", "clock_frac_under_30s",
        ]:
            own_col = f"{own}_{suffix}"
            opp_col = f"{opp}_{suffix}"
            if own_col in game_df:
                row[f"own_{suffix}"] = game_df[own_col]
            if opp_col in game_df:
                row[f"opp_{suffix}"] = game_df[opp_col]

        if "eval_final_pawns" in game_df:
            row["own_eval_final_pawns"] = sign * game_df["eval_final_pawns"]

        # Board-aware perspective features from chess_board_features_workflow.py.
        board_map = {
            "own_board_checks_given": f"checks_given_{own}",
            "opp_board_checks_given": f"checks_given_{opp}",
            "own_board_check_density": f"check_density_{own}",
            "opp_board_check_density": f"check_density_{opp}",
            "own_first_capture_move": f"first_capture_move_{own}",
            "opp_first_capture_move": f"first_capture_move_{opp}",
            "own_castle_move": f"castle_move_{own}",
            "own_board_castle_side": f"castle_side_{own}",
            "own_consec_same_piece": f"consec_same_piece_{own}",
            "own_legal_moves_move5": f"legal_moves_{own}_move5",
            "own_acpl": f"acpl_{own}",
            "own_inaccuracy_count": f"inaccuracy_count_{own}",
            "own_mistake_count": f"mistake_count_{own}",
            "own_blunder_count": f"blunder_count_{own}",
            "own_blunder_density": f"blunder_density_{own}",
        }
        for out_col, in_col in board_map.items():
            if in_col in game_df:
                row[out_col] = game_df[in_col]

        territory_col = "white_territory_depth" if own == "white" else "black_territory_depth"
        if territory_col in game_df:
            row["own_territory_depth"] = game_df[territory_col]

        if "material_balance_end" in game_df:
            row["own_material_balance_end"] = sign * game_df["material_balance_end"]

        player_rows.append(row)

    out = pd.concat(player_rows, ignore_index=True)

    out["own_castled"] = out["own_castled"].fillna(False).astype(bool)
    out["opp_castled"] = out["opp_castled"].fillna(False).astype(bool)
    out["own_castle_kingside"] = (out["own_castle_side"] == "kingside").astype(int)
    out["own_castle_queenside"] = (out["own_castle_side"] == "queenside").astype(int)

    out["own_capture_rate"] = _base.safe_divide(out["own_captures"], out["own_moves"])
    out["opp_capture_rate"] = _base.safe_divide(out["opp_captures"], out["opp_moves"])
    out["own_check_rate"] = _base.safe_divide(out["own_checks"], out["own_moves"])
    out["opp_check_rate"] = _base.safe_divide(out["opp_checks"], out["opp_moves"])
    out["game_length_moves"] = out["num_halfmoves"] / 2

    return out


def parse_to_player_data(
    raw_df: pd.DataFrame,
    *,
    min_games: int = 20,
    include_distribution_features: bool = False,
    include_board_features: bool = False,
    board_moves_col: str | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Build one row per player for player-archetype clustering.

    ``include_board_features=False`` is the fast default. Use
    ``include_board_features=True`` when you want the friend-code style
    board-aware summary features.
    """
    if not include_board_features:
        return _base.parse_to_player_data(
            raw_df,
            min_games=min_games,
            include_distribution_features=include_distribution_features,
        )

    game_df = parse_to_game_data(
        raw_df,
        include_text_columns=True,
        include_board_features=True,
        board_moves_col=board_moves_col,
        n_jobs=n_jobs,
    )
    player_df = _make_player_game_frame(game_df)

    agg_spec = {
        "n_games": ("game_id", "count"),
        "median_elo": ("elo", "median"),
        "mean_elo": ("elo", "mean"),
        "elo_std": ("elo", "std"),
        "median_opponent_elo": ("opponent_elo", "median"),
        "mean_opponent_elo": ("opponent_elo", "mean"),
        "mean_result": ("result_score", "mean"),
        "white_rate": ("is_white", "mean"),
        "mean_game_length": ("game_length_moves", "mean"),
        "median_game_length": ("game_length_moves", "median"),
        "mean_initial_seconds": ("initial_seconds", "mean"),
        "mean_increment_seconds": ("increment_seconds", "mean"),
        "capture_rate": ("own_capture_rate", "mean"),
        "opponent_capture_rate": ("opp_capture_rate", "mean"),
        "check_rate": ("own_check_rate", "mean"),
        "opponent_check_rate": ("opp_check_rate", "mean"),
        "castle_rate": ("own_castled", "mean"),
        "kingside_castle_rate": ("own_castle_kingside", "mean"),
        "queenside_castle_rate": ("own_castle_queenside", "mean"),
        "first_move_diversity": ("first_own_move", "nunique"),
        "eco_diversity": ("eco", "nunique"),
        "opening_diversity": ("opening", "nunique"),
    }

    optional_agg = {
        "mean_own_clock_final_seconds": ("own_clock_last_seconds", "mean"),
        "mean_own_clock_min_seconds": ("own_clock_min_seconds", "mean"),
        "mean_own_clock_frac_under_10s": ("own_clock_frac_under_10s", "mean"),
        "mean_own_clock_frac_under_30s": ("own_clock_frac_under_30s", "mean"),
        "mean_own_clock_drop_observed_seconds": ("own_clock_drop_observed_seconds", "mean"),
        "mean_own_eval_final_pawns": ("own_eval_final_pawns", "mean"),
        "mean_abs_eval_pawns": ("eval_mean_abs_pawns", "mean"),
        "mean_eval_swing_pawns": ("eval_swing_pawns", "mean"),
        "mean_balanced_eval_frac": ("balanced_eval_frac", "mean"),
        "mean_own_acpl": ("own_acpl", "mean"),
        "mean_own_blunder_density": ("own_blunder_density", "mean"),
        "mean_own_inaccuracy_count": ("own_inaccuracy_count", "mean"),
        "mean_own_mistake_count": ("own_mistake_count", "mean"),
        "mean_own_blunder_count": ("own_blunder_count", "mean"),
        "mean_own_territory_depth": ("own_territory_depth", "mean"),
        "mean_own_legal_moves_move5": ("own_legal_moves_move5", "mean"),
        "mean_own_consec_same_piece": ("own_consec_same_piece", "mean"),
        "mean_own_first_capture_move": ("own_first_capture_move", "mean"),
        "mean_own_material_balance_end": ("own_material_balance_end", "mean"),
    }
    for out_col, (in_col, reducer) in optional_agg.items():
        if in_col in player_df:
            agg_spec[out_col] = (in_col, reducer)

    out = player_df.groupby("player").agg(**agg_spec)
    out["elo_std"] = out["elo_std"].fillna(0.0)
    out["first_move_diversity_rate"] = out["first_move_diversity"] / out["n_games"]
    out["eco_diversity_rate"] = out["eco_diversity"] / out["n_games"]
    out["opening_diversity_rate"] = out["opening_diversity"] / out["n_games"]

    entropy_df = player_df.groupby("player").agg(
        eco_entropy=("eco", _base.entropy),
        opening_entropy=("opening", _base.entropy),
        speed_entropy=("speed", _base.entropy),
    )
    out = out.join(entropy_df)

    if include_distribution_features:
        if "speed" in player_df:
            out = out.join(pd.crosstab(player_df["player"], player_df["speed"], normalize="index").add_prefix("speed_frac_"), how="left")
        if "eco_family" in player_df:
            out = out.join(pd.crosstab(player_df["player"], player_df["eco_family"], normalize="index").add_prefix("eco_family_frac_"), how="left")
        if "first_own_move" in player_df:
            out = out.join(pd.crosstab(player_df["player"], player_df["first_own_move"], normalize="index").add_prefix("first_move_frac_"), how="left")

    return out.fillna(0.0).query("n_games >= @min_games").reset_index()


def select_player_summary_features(
    player_df: pd.DataFrame,
    *,
    include_player: bool = True,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select the compact Feature Set 1 table for clustering."""
    if feature_columns is None:
        board_cols_present = any(column in player_df for column in PLAYER_SUMMARY_FEATURES_BOARD_AWARE[-7:])
        feature_columns = PLAYER_SUMMARY_FEATURES_BOARD_AWARE if board_cols_present else PLAYER_SUMMARY_FEATURES_FAST

    missing = [column for column in feature_columns if column not in player_df]
    if missing:
        raise ValueError(f"Player summary table is missing expected feature columns: {missing}")

    columns = (["player"] if include_player and "player" in player_df else []) + list(feature_columns)
    return player_df[columns].copy()


def format_table(input_df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """Backward-compatible wrapper: type1=game data, type3=player data."""
    table_type = table_type.lower().strip()
    if table_type == "type1":
        return parse_to_game_data(input_df)
    if table_type == "type2":
        raise ValueError("type2/player-game output has been removed. Use parse_to_player_data(...) instead.")
    if table_type == "type3":
        return parse_to_player_data(input_df)
    raise ValueError("Invalid table_type. Use 'type1' or 'type3'.")
