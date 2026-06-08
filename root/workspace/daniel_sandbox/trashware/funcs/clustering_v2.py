from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
import re

import numpy as np
import pandas as pd


PIECE_LETTERS = frozenset("KQRBN")
PROMOTION_RE = re.compile(r"=([QRBN])")
LIST_SEP = "|"


# ---------------------------------------------------------------------------
# Generic utilities.
# ---------------------------------------------------------------------------

def require(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Raise a readable error if required columns are missing."""
    missing = [column for column in columns if column not in df]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def fill_missing(df: pd.DataFrame, defaults: dict[str, object]) -> pd.DataFrame:
    """Add missing columns with scalar defaults."""
    for column, value in defaults.items():
        if column not in df:
            df[column] = value
    return df


def safe_divide(a: object, b: object) -> np.ndarray:
    """Vectorized division that returns NaN where the denominator is zero."""
    denominator = pd.to_numeric(b, errors="coerce")
    numerator = pd.to_numeric(a, errors="coerce")
    return np.where(denominator == 0, np.nan, numerator / denominator)


def entropy(series: pd.Series) -> float:
    """Shannon entropy of a categorical series."""
    p = series.dropna().value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_number_list(value: object) -> list[float | None]:
    """
    Parse a serialized pipe-separated numeric list.

    Empty elements are kept as ``None`` so ply alignment is preserved.
    """
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        parsed = []
        for item in value:
            if _is_missing(item) or item == "":
                parsed.append(None)
            else:
                try:
                    parsed.append(float(item))
                except (TypeError, ValueError):
                    parsed.append(None)
        return parsed

    text = str(value)
    if text == "":
        return []

    parsed = []
    for item in text.split(LIST_SEP):
        if item == "":
            parsed.append(None)
        else:
            try:
                parsed.append(float(item))
            except ValueError:
                parsed.append(None)
    return parsed


def result_to_white_score(df: pd.DataFrame) -> pd.Series:
    """Infer result score from White's perspective."""
    if "result_white_score" in df:
        return pd.to_numeric(df["result_white_score"], errors="coerce")

    if "result" in df:
        result = df["result"].astype(str).str.lower().str.strip().str.replace("½", "1/2")
        return result.map({
            "1-0": 1.0,
            "0-1": 0.0,
            "1/2-1/2": 0.5,
            "draw": 0.5,
            "white": 1.0,
            "black": 0.0,
        })

    if "winner" in df:
        winner = df["winner"].astype(str).str.lower().str.strip()
        return winner.map({
            "white": 1.0,
            "black": 0.0,
            "draw": 0.5,
        })

    raise ValueError("Could not infer result_white_score. Need 'result_white_score', 'result', or 'winner'.")


# ---------------------------------------------------------------------------
# SAN-derived features.
# ---------------------------------------------------------------------------

def san_tokens_from_string(moves_san: object) -> list[str]:
    """Split a normalized SAN string into tokens."""
    if _is_missing(moves_san):
        return []
    return [token for token in str(moves_san).split() if token]


def extract_san_features(tokens: Sequence[str]) -> dict[str, object]:
    """
    Extract syntactic chess features from SAN tokens.

    These do not reconstruct board state. They are fast string features, suitable
    for large database-level clustering work.
    """
    num_halfmoves = len(tokens)

    features: dict[str, object] = {
        "num_halfmoves": num_halfmoves,
        "num_moves": num_halfmoves / 2,
        "num_fullmoves": (num_halfmoves + 1) // 2,
        "num_white_moves": (num_halfmoves + 1) // 2,
        "num_black_moves": num_halfmoves // 2,
        "num_captures": 0,
        "white_captures": 0,
        "black_captures": 0,
        "num_checks": 0,
        "white_checks": 0,
        "black_checks": 0,
        "num_checkmates": 0,
        "white_checkmates": 0,
        "black_checkmates": 0,
        "num_promotions": 0,
        "num_queen_promotions": 0,
        "num_rook_promotions": 0,
        "num_bishop_promotions": 0,
        "num_knight_promotions": 0,
        "num_castles_kingside": 0,
        "num_castles_queenside": 0,
        "white_castled": False,
        "black_castled": False,
        "white_castle_side": None,
        "black_castle_side": None,
        "num_pawn_moves": 0,
        "num_piece_moves": 0,
        "num_king_moves": 0,
        "num_queen_moves": 0,
        "num_rook_moves": 0,
        "num_bishop_moves": 0,
        "num_knight_moves": 0,
        "first_white_move": tokens[0] if num_halfmoves >= 1 else None,
        "first_black_move": tokens[1] if num_halfmoves >= 2 else None,
        "last_move": tokens[-1] if num_halfmoves else None,
        "opening_san_4ply": " ".join(tokens[:4]),
        "opening_san_6ply": " ".join(tokens[:6]),
        "opening_san_10ply": " ".join(tokens[:10]),
    }

    for ply, token in enumerate(tokens):
        is_white = (ply % 2) == 0

        is_queenside_castle = token.startswith("O-O-O") or token.startswith("0-0-0")
        is_kingside_castle = (
            not is_queenside_castle
            and (token.startswith("O-O") or token.startswith("0-0"))
        )

        if is_queenside_castle or is_kingside_castle:
            if is_queenside_castle:
                features["num_castles_queenside"] += 1
                side = "queenside"
            else:
                features["num_castles_kingside"] += 1
                side = "kingside"

            if is_white:
                features["white_castled"] = True
                features["white_castle_side"] = side
            else:
                features["black_castled"] = True
                features["black_castle_side"] = side
        else:
            first = token[0]
            if first in PIECE_LETTERS:
                features["num_piece_moves"] += 1
                piece_column = {
                    "K": "num_king_moves",
                    "Q": "num_queen_moves",
                    "R": "num_rook_moves",
                    "B": "num_bishop_moves",
                    "N": "num_knight_moves",
                }[first]
                features[piece_column] += 1
            else:
                features["num_pawn_moves"] += 1

        if "x" in token:
            features["num_captures"] += 1
            features["white_captures" if is_white else "black_captures"] += 1

        if "+" in token:
            features["num_checks"] += 1
            features["white_checks" if is_white else "black_checks"] += 1

        if "#" in token:
            features["num_checkmates"] += 1
            features["white_checkmates" if is_white else "black_checkmates"] += 1

        promotion_match = PROMOTION_RE.search(token)
        if promotion_match:
            features["num_promotions"] += 1
            promotion_column = {
                "Q": "num_queen_promotions",
                "R": "num_rook_promotions",
                "B": "num_bishop_promotions",
                "N": "num_knight_promotions",
            }[promotion_match.group(1)]
            features[promotion_column] += 1

    return features


def build_san_feature_frame(moves_san: pd.Series) -> pd.DataFrame:
    """Build a dataframe of SAN-derived features from a ``moves_san`` series."""
    records = [extract_san_features(san_tokens_from_string(value)) for value in moves_san]
    return pd.DataFrame.from_records(records, index=moves_san.index)


# ---------------------------------------------------------------------------
# Clock/eval-derived features.
# ---------------------------------------------------------------------------

def _numeric_array(values: Iterable[float | None]) -> np.ndarray:
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)


def _safe_stat(values: np.ndarray, reducer: str) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    if reducer == "first":
        return float(valid[0])
    if reducer == "last":
        return float(valid[-1])
    if reducer == "min":
        return float(np.min(valid))
    if reducer == "max":
        return float(np.max(valid))
    if reducer == "mean":
        return float(np.mean(valid))
    if reducer == "std":
        return float(np.std(valid))
    raise ValueError(f"Unknown reducer: {reducer}")


def extract_clock_features(clock_values: Sequence[float | None], num_halfmoves: int | None = None) -> dict[str, float]:
    """Extract per-game clock features from per-ply clock seconds."""
    values = _numeric_array(clock_values)
    if num_halfmoves is None:
        num_halfmoves = len(values)

    white = values[0::2]
    black = values[1::2]
    num_clock = int(np.count_nonzero(~np.isnan(values)))

    features: dict[str, float] = {
        "num_clock_annotations": num_clock,
        "clock_coverage": num_clock / num_halfmoves if num_halfmoves else 0.0,
    }

    for color, color_values in {"white": white, "black": black}.items():
        first = _safe_stat(color_values, "first")
        last = _safe_stat(color_values, "last")
        features[f"{color}_clock_first_seconds"] = first
        features[f"{color}_clock_last_seconds"] = last
        features[f"{color}_clock_min_seconds"] = _safe_stat(color_values, "min")
        features[f"{color}_clock_max_seconds"] = _safe_stat(color_values, "max")
        features[f"{color}_clock_mean_seconds"] = _safe_stat(color_values, "mean")
        features[f"{color}_clock_std_seconds"] = _safe_stat(color_values, "std")
        features[f"{color}_clock_drop_observed_seconds"] = first - last if not np.isnan(first) and not np.isnan(last) else np.nan

        valid = color_values[~np.isnan(color_values)]
        features[f"{color}_clock_frac_under_10s"] = float(np.mean(valid < 10)) if len(valid) else np.nan
        features[f"{color}_clock_frac_under_30s"] = float(np.mean(valid < 30)) if len(valid) else np.nan
        features[f"{color}_clock_frac_under_60s"] = float(np.mean(valid < 60)) if len(valid) else np.nan

    return features


def extract_eval_features(eval_values: Sequence[float | None], num_halfmoves: int | None = None) -> dict[str, float]:
    """Extract per-game engine-eval features from White-perspective evals."""
    values = _numeric_array(eval_values)
    if num_halfmoves is None:
        num_halfmoves = len(values)

    valid = values[~np.isnan(values)]
    num_eval = int(len(valid))

    if num_eval == 0:
        return {
            "num_eval_annotations": 0,
            "eval_coverage": 0.0,
            "eval_initial_pawns": np.nan,
            "eval_final_pawns": np.nan,
            "eval_mean_pawns": np.nan,
            "eval_mean_abs_pawns": np.nan,
            "eval_max_abs_pawns": np.nan,
            "eval_std_pawns": np.nan,
            "eval_swing_pawns": np.nan,
            "white_advantage_frac": np.nan,
            "black_advantage_frac": np.nan,
            "balanced_eval_frac": np.nan,
        }

    return {
        "num_eval_annotations": num_eval,
        "eval_coverage": num_eval / num_halfmoves if num_halfmoves else 0.0,
        "eval_initial_pawns": float(valid[0]),
        "eval_final_pawns": float(valid[-1]),
        "eval_mean_pawns": float(np.mean(valid)),
        "eval_mean_abs_pawns": float(np.mean(np.abs(valid))),
        "eval_max_abs_pawns": float(np.max(np.abs(valid))),
        "eval_std_pawns": float(np.std(valid)),
        "eval_swing_pawns": float(np.nanmax(values) - np.nanmin(values)),
        "white_advantage_frac": float(np.mean(valid > 1.0)),
        "black_advantage_frac": float(np.mean(valid < -1.0)),
        "balanced_eval_frac": float(np.mean(np.abs(valid) <= 1.0)),
    }


def build_annotation_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build clock/eval feature columns from serialized per-ply annotation columns."""
    index = df.index

    if "clock_seconds_by_ply" in df:
        clock_records = []
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            clocks = parse_number_list(row_dict.get("clock_seconds_by_ply"))
            n = row_dict.get("num_halfmoves")
            n_int = int(n) if not _is_missing(n) else len(clocks)
            clock_records.append(extract_clock_features(clocks, n_int))
        clock_df = pd.DataFrame.from_records(clock_records, index=index)
    else:
        clock_df = pd.DataFrame(index=index)

    if "eval_pawns_by_ply" in df:
        eval_records = []
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            evals = parse_number_list(row_dict.get("eval_pawns_by_ply"))
            n = row_dict.get("num_halfmoves")
            n_int = int(n) if not _is_missing(n) else len(evals)
            eval_records.append(extract_eval_features(evals, n_int))
        eval_df = pd.DataFrame.from_records(eval_records, index=index)
    else:
        eval_df = pd.DataFrame(index=index)

    return pd.concat([clock_df, eval_df], axis=1)


# ---------------------------------------------------------------------------
# Public table builders.
# ---------------------------------------------------------------------------

def parse_to_game_data(raw_df: pd.DataFrame, *, include_text_columns: bool = True) -> pd.DataFrame:
    """
    Build one ML-oriented row per game.

    This is the replacement for the old ``format_table(df, "type1")`` path.
    It keeps metadata and adds SAN, clock and eval-derived features.
    """
    output_df = raw_df.copy().reset_index(drop=True)

    if "game_id" not in output_df:
        output_df.insert(0, "game_id", np.arange(len(output_df), dtype=int))

    require(output_df, ["white", "black", "white_elo", "black_elo"], "Game table")

    output_df["result_white_score"] = result_to_white_score(output_df)

    if "moves_san" in output_df:
        san_features = build_san_feature_frame(output_df["moves_san"])
        # Parser-supplied columns are overwritten by the SAN-derived values so
        # the table stays consistent if moves_san was edited or regenerated.
        for column in san_features.columns:
            output_df[column] = san_features[column]
    elif "num_halfmoves" not in output_df:
        raise ValueError("Game table needs either 'moves_san' or 'num_halfmoves'.")

    output_df["num_halfmoves"] = pd.to_numeric(output_df["num_halfmoves"], errors="coerce").fillna(0).astype(int)

    if "num_white_moves" not in output_df:
        output_df["num_white_moves"] = ((output_df["num_halfmoves"] + 1) // 2).astype(int)
    if "num_black_moves" not in output_df:
        output_df["num_black_moves"] = (output_df["num_halfmoves"] // 2).astype(int)
    if "num_fullmoves" not in output_df:
        output_df["num_fullmoves"] = ((output_df["num_halfmoves"] + 1) // 2).astype(int)
    if "num_moves" not in output_df:
        output_df["num_moves"] = output_df["num_halfmoves"] / 2

    annotation_features = build_annotation_feature_frame(output_df)
    for column in annotation_features.columns:
        output_df[column] = annotation_features[column]

    output_df = fill_missing(output_df, {
        "event": None,
        "site": None,
        "date": None,
        "time": None,
        "speed": None,
        "initial_seconds": np.nan,
        "increment_seconds": np.nan,
        "termination": None,
        "eco": None,
        "eco_family": None,
        "opening": None,
        "first_white_move": None,
        "first_black_move": None,
        "opening_san_4ply": None,
        "opening_san_6ply": None,
        "opening_san_10ply": None,
        "white_captures": 0,
        "black_captures": 0,
        "white_checks": 0,
        "black_checks": 0,
        "white_checkmates": 0,
        "black_checkmates": 0,
        "white_castled": False,
        "black_castled": False,
        "white_castle_side": None,
        "black_castle_side": None,
        "white_rating_diff": np.nan,
        "black_rating_diff": np.nan,
    })

    output_df["white_elo"] = pd.to_numeric(output_df["white_elo"], errors="coerce")
    output_df["black_elo"] = pd.to_numeric(output_df["black_elo"], errors="coerce")
    output_df["avg_elo"] = (output_df["white_elo"] + output_df["black_elo"]) / 2
    output_df["elo_diff"] = output_df["white_elo"] - output_df["black_elo"]
    output_df["abs_elo_diff"] = output_df["elo_diff"].abs()

    if not include_text_columns:
        drop_columns = [
            "event", "site", "opening", "moves_san", "clock_seconds_by_ply",
            "eval_pawns_by_ply", "eval_mate_by_ply",
        ]
        output_df = output_df.drop(columns=[column for column in drop_columns if column in output_df])

    return output_df


def parse_to_player_game_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build two rows per game, one from each player's perspective.

    This is useful as the intermediate table for player archetype aggregation.
    """
    game_df = parse_to_game_data(raw_df)

    common_cols = [
        "game_id", "event", "site", "date", "time", "speed", "initial_seconds",
        "increment_seconds", "termination", "eco", "eco_family", "opening",
        "opening_san_4ply", "opening_san_6ply", "opening_san_10ply", "num_halfmoves",
        "eval_initial_pawns", "eval_final_pawns", "eval_mean_pawns",
        "eval_mean_abs_pawns", "eval_max_abs_pawns", "eval_std_pawns",
        "eval_swing_pawns", "white_advantage_frac", "black_advantage_frac",
        "balanced_eval_frac",
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
            "clock_first_seconds", "clock_last_seconds", "clock_min_seconds",
            "clock_max_seconds", "clock_mean_seconds", "clock_std_seconds",
            "clock_drop_observed_seconds", "clock_frac_under_10s",
            "clock_frac_under_30s", "clock_frac_under_60s",
        ]:
            own_col = f"{own}_{suffix}"
            opp_col = f"{opp}_{suffix}"
            if own_col in game_df:
                row[f"own_{suffix}"] = game_df[own_col]
            if opp_col in game_df:
                row[f"opp_{suffix}"] = game_df[opp_col]

        for column in ["eval_initial_pawns", "eval_final_pawns", "eval_mean_pawns"]:
            if column in game_df:
                row[f"own_{column}"] = sign * game_df[column]

        player_rows.append(row)

    output_df = pd.concat(player_rows, ignore_index=True)

    output_df["own_castled"] = output_df["own_castled"].fillna(False).astype(bool)
    output_df["opp_castled"] = output_df["opp_castled"].fillna(False).astype(bool)
    output_df["own_castle_kingside"] = (output_df["own_castle_side"] == "kingside").astype(int)
    output_df["own_castle_queenside"] = (output_df["own_castle_side"] == "queenside").astype(int)

    output_df["own_capture_rate"] = safe_divide(output_df["own_captures"], output_df["own_moves"])
    output_df["opp_capture_rate"] = safe_divide(output_df["opp_captures"], output_df["opp_moves"])
    output_df["own_check_rate"] = safe_divide(output_df["own_checks"], output_df["own_moves"])
    output_df["opp_check_rate"] = safe_divide(output_df["opp_checks"], output_df["opp_moves"])
    output_df["game_length_moves"] = output_df["num_halfmoves"] / 2

    return output_df


def parse_to_player_data(
    raw_df: pd.DataFrame,
    *,
    min_games: int = 20,
    include_speed_distribution: bool = True,
    include_eco_distribution: bool = True,
    include_first_move_distribution: bool = True,
) -> pd.DataFrame:
    """
    Build one row per player for player-archetype clustering.

    This is the replacement for the old ``format_table(df, "type3")`` path.
    """
    player_df = parse_to_player_game_data(raw_df)

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
        "mean_opp_clock_frac_under_10s": ("opp_clock_frac_under_10s", "mean"),
        "mean_own_eval_final_pawns": ("own_eval_final_pawns", "mean"),
        "mean_own_eval_mean_pawns": ("own_eval_mean_pawns", "mean"),
        "mean_abs_eval_pawns": ("eval_mean_abs_pawns", "mean"),
        "mean_eval_swing_pawns": ("eval_swing_pawns", "mean"),
    }

    for output_column, (input_column, reducer) in optional_agg.items():
        if input_column in player_df:
            agg_spec[output_column] = (input_column, reducer)

    output_df = player_df.groupby("player").agg(**agg_spec)

    output_df["elo_std"] = output_df["elo_std"].fillna(0.0)
    output_df["first_move_diversity_rate"] = output_df["first_move_diversity"] / output_df["n_games"]
    output_df["eco_diversity_rate"] = output_df["eco_diversity"] / output_df["n_games"]
    output_df["opening_diversity_rate"] = output_df["opening_diversity"] / output_df["n_games"]

    entropy_df = player_df.groupby("player").agg(
        eco_entropy=("eco", entropy),
        opening_entropy=("opening", entropy),
        speed_entropy=("speed", entropy),
    )
    output_df = output_df.join(entropy_df)

    if include_speed_distribution and "speed" in player_df:
        speed_df = pd.crosstab(player_df["player"], player_df["speed"], normalize="index").add_prefix("speed_frac_")
        output_df = output_df.join(speed_df, how="left")

    if include_eco_distribution and "eco_family" in player_df:
        eco_df = pd.crosstab(player_df["player"], player_df["eco_family"], normalize="index").add_prefix("eco_family_frac_")
        output_df = output_df.join(eco_df, how="left")

    if include_first_move_distribution and "first_own_move" in player_df:
        first_move_df = pd.crosstab(player_df["player"], player_df["first_own_move"], normalize="index").add_prefix("first_move_frac_")
        output_df = output_df.join(first_move_df, how="left")

    output_df = (
        output_df
        .fillna(0.0)
        .query("n_games >= @min_games")
        .reset_index()
    )

    return output_df


def format_table(input_df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """
    Backward-compatible wrapper around the new modular table builders.

    ``type1`` -> ``parse_to_game_data``
    ``type2`` -> ``parse_to_player_game_data``
    ``type3`` -> ``parse_to_player_data``
    """
    table_type = table_type.lower().strip()

    if table_type == "type1":
        return parse_to_game_data(input_df)
    if table_type == "type2":
        return parse_to_player_game_data(input_df)
    if table_type == "type3":
        return parse_to_player_data(input_df)

    raise ValueError(f"Invalid table_type: {table_type}. Must be 'type1', 'type2', or 'type3'.")
