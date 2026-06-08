from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

import numpy as np
import pandas as pd

LIST_SEP = "|"
PIECES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
PROMOTION_RE = re.compile(r"=([QRBN])")
DEST_RE = re.compile(r"([a-h][1-8])")

# Compact, human-readable feature set for clustering. Elo/result columns are not here.
SUMMARY_FEATURES = [
    "avg_game_length",
    "capture_rate",
    "check_rate",
    "castle_rate",
    "queenside_castle_rate",
    "pawn_move_rate",
    "knight_move_rate",
    "bishop_move_rate",
    "rook_move_rate",
    "queen_move_rate",
    "king_move_rate",
    "promotion_rate",
    "queen_promotion_rate",
    "first_capture_move_mean",
    "early_queen_move_rate",
    "first_move_diversity_rate",
    "territory_depth_mean",
    "mean_move_seconds",
    "time_pressure_30s_rate",
]

NON_TRAIN_COLUMNS = {
    "player", "elo_mean", "elo_median", "elo_std", "n_games", "win_rate",
    "mean_result", "rating_diff_mean", "opponent_elo_mean", "opponent_elo_median",
}


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_number_list(value: object) -> list[float | None]:
    if _missing(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        values = value
    else:
        values = str(value).split(LIST_SEP) if str(value) else []
    out: list[float | None] = []
    for item in values:
        if _missing(item) or item == "":
            out.append(None)
        else:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(None)
    return out


def _san_tokens(value: object) -> list[str]:
    if _missing(value):
        return []
    return [token for token in str(value).split() if token]


def _piece_from_san(token: str) -> str:
    if token.startswith(("O-O", "0-0")):
        return "king"
    return {"N": "knight", "B": "bishop", "R": "rook", "Q": "queen", "K": "king"}.get(token[:1], "pawn")


def _destination_rank(token: str) -> int | None:
    clean = token.replace("+", "").replace("#", "")
    clean = PROMOTION_RE.sub("", clean)
    matches = DEST_RE.findall(clean)
    if not matches:
        return None
    return int(matches[-1][1])


def _empty_game_features() -> dict[str, object]:
    out: dict[str, object] = {
        "num_halfmoves": 0,
        "num_moves": 0.0,
        "first_white_move": None,
        "first_black_move": None,
        "opening_san_4ply": None,
        "opening_san_6ply": None,
        "opening_san_10ply": None,
    }
    for color in ["white", "black"]:
        out.update({
            f"{color}_moves": 0,
            f"{color}_captures": 0,
            f"{color}_checks": 0,
            f"{color}_promotions": 0,
            f"{color}_queen_promotions": 0,
            f"{color}_castled": False,
            f"{color}_castle_side": None,
            f"{color}_first_capture_move": np.nan,
            f"{color}_early_queen_moves": 0,
            f"{color}_territory_depth_max": 0.0,
        })
        for piece in PIECES:
            out[f"{color}_{piece}_moves"] = 0
    return out


def extract_san_game_features(moves_san: object) -> dict[str, object]:
    tokens = _san_tokens(moves_san)
    if not tokens:
        return _empty_game_features()

    out = _empty_game_features()
    out.update({
        "num_halfmoves": len(tokens),
        "num_moves": len(tokens) / 2,
        "first_white_move": tokens[0] if len(tokens) >= 1 else None,
        "first_black_move": tokens[1] if len(tokens) >= 2 else None,
        "opening_san_4ply": " ".join(tokens[:4]),
        "opening_san_6ply": " ".join(tokens[:6]),
        "opening_san_10ply": " ".join(tokens[:10]),
    })

    for ply, token in enumerate(tokens):
        color = "white" if ply % 2 == 0 else "black"
        fullmove = ply // 2 + 1
        piece = _piece_from_san(token)
        out[f"{color}_moves"] += 1
        out[f"{color}_{piece}_moves"] += 1

        if "x" in token:
            out[f"{color}_captures"] += 1
            if pd.isna(out[f"{color}_first_capture_move"]):
                out[f"{color}_first_capture_move"] = fullmove
        if "+" in token or "#" in token:
            out[f"{color}_checks"] += 1
        if token.startswith(("O-O-O", "0-0-0")):
            out[f"{color}_castled"] = True
            out[f"{color}_castle_side"] = "queenside"
        elif token.startswith(("O-O", "0-0")):
            out[f"{color}_castled"] = True
            out[f"{color}_castle_side"] = "kingside"

        promo = PROMOTION_RE.search(token)
        if promo:
            out[f"{color}_promotions"] += 1
            if promo.group(1) == "Q":
                out[f"{color}_queen_promotions"] += 1

        if piece == "queen" and fullmove <= 10:
            out[f"{color}_early_queen_moves"] += 1

        rank = _destination_rank(token)
        if rank is not None:
            depth = max(0, rank - 4) if color == "white" else max(0, 5 - rank)
            out[f"{color}_territory_depth_max"] = max(out[f"{color}_territory_depth_max"], float(depth))

    return out


def build_san_feature_frame(moves_san: pd.Series) -> pd.DataFrame:
    return pd.DataFrame.from_records([extract_san_game_features(x) for x in moves_san], index=moves_san.index)


def _result_to_white_score(df: pd.DataFrame) -> pd.Series:
    if "result_white_score" in df:
        return pd.to_numeric(df["result_white_score"], errors="coerce")
    result = df.get("result", pd.Series(index=df.index, dtype=object)).astype(str).str.replace("½", "1/2")
    return result.map({"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5})


def _clock_game_features(row: pd.Series, color: str) -> dict[str, float]:
    clocks = parse_number_list(row.get("clock_seconds_by_ply"))
    if not clocks:
        return {"mean_move_seconds": np.nan, "time_pressure_30s_rate": np.nan}
    initial = pd.to_numeric(row.get("initial_seconds"), errors="coerce")
    increment = pd.to_numeric(row.get("increment_seconds"), errors="coerce")
    if pd.isna(initial):
        initial = np.nan
    if pd.isna(increment):
        increment = 0.0

    vals = np.array([np.nan if x is None else float(x) for x in (clocks[0::2] if color == "white" else clocks[1::2])], dtype=float)
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return {"mean_move_seconds": np.nan, "time_pressure_30s_rate": np.nan}

    prev = np.r_[initial, valid[:-1]] if not pd.isna(initial) else np.r_[np.nan, valid[:-1]]
    spent = prev + increment - valid
    spent = spent[(~np.isnan(spent)) & (spent >= 0) & (spent < 600)]
    return {
        "mean_move_seconds": float(np.mean(spent)) if len(spent) else np.nan,
        "time_pressure_30s_rate": float(np.mean(valid < 30)),
    }


def filter_rated_standard_10_0_rapid(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rated standard games with TimeControl 600+0. This is 10+0 rapid."""
    out = df.copy()
    if "initial_seconds" not in out or "increment_seconds" not in out:
        if "time_control" in out:
            parts = out["time_control"].astype(str).str.extract(r"^(\d+)\+(\d+)$")
            out["initial_seconds"] = pd.to_numeric(parts[0], errors="coerce")
            out["increment_seconds"] = pd.to_numeric(parts[1], errors="coerce")
    mask = (pd.to_numeric(out.get("initial_seconds"), errors="coerce") == 600) & (pd.to_numeric(out.get("increment_seconds"), errors="coerce") == 0)
    if "rated" in out:
        mask &= out["rated"].fillna(True).astype(bool)
    elif "event" in out:
        mask &= out["event"].astype(str).str.lower().str.startswith("rated")
    if "standard" in out:
        mask &= out["standard"].fillna(True).astype(bool)
    elif "variant" in out:
        mask &= out["variant"].isna() | out["variant"].isin(["", "Standard"])
    return out.loc[mask].reset_index(drop=True)


def parse_to_game_data(raw_df: pd.DataFrame, *, include_board_features: bool = False, board_moves_col: str | None = None, n_jobs: int = -1) -> pd.DataFrame:
    df = raw_df.copy().reset_index(drop=True)
    if "game_id" not in df:
        df.insert(0, "game_id", np.arange(len(df), dtype=int))
    if "result_white_score" not in df:
        df["result_white_score"] = _result_to_white_score(df)
    if "moves_san" in df:
        san_df = build_san_feature_frame(df["moves_san"])
        for col in san_df.columns:
            df[col] = san_df[col]

    for col in ["white_elo", "black_elo", "initial_seconds", "increment_seconds"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if include_board_features:
        from .board_features import extract_board_features_dataframe
        board_df = extract_board_features_dataframe(df, moves_col=board_moves_col, result_col="result" if "result" in df else None, n_jobs=n_jobs)
        overlap = [c for c in board_df.columns if c in df.columns]
        board_df = board_df.rename(columns={c: f"board_{c}" for c in overlap})
        df = pd.concat([df.reset_index(drop=True), board_df.reset_index(drop=True)], axis=1)

    return df


def parse_to_player_game_data(raw_df: pd.DataFrame, *, include_board_features: bool = False, board_moves_col: str | None = None, n_jobs: int = -1) -> pd.DataFrame:
    g = parse_to_game_data(raw_df, include_board_features=include_board_features, board_moves_col=board_moves_col, n_jobs=n_jobs)
    rows: list[pd.DataFrame] = []

    common_cols = ["game_id", "date", "time", "time_control", "speed", "initial_seconds", "increment_seconds", "eco", "opening", "opening_san_4ply", "opening_san_6ply", "opening_san_10ply", "num_halfmoves", "num_moves"]
    common = {c: g[c] for c in common_cols if c in g}

    for color, opp in [("white", "black"), ("black", "white")]:
        other_score = 1.0 - g["result_white_score"] if color == "black" else g["result_white_score"]
        row = pd.DataFrame({
            **common,
            "player": g[color],
            "opponent": g[opp],
            "color": color,
            "is_white": int(color == "white"),
            "elo": g[f"{color}_elo"],
            "opponent_elo": g[f"{opp}_elo"],
            "result_score": other_score,
            "rating_diff": g.get(f"{color}_rating_diff", np.nan),
            "own_moves": g[f"{color}_moves"],
            "own_captures": g[f"{color}_captures"],
            "own_checks": g[f"{color}_checks"],
            "own_promotions": g[f"{color}_promotions"],
            "own_queen_promotions": g[f"{color}_queen_promotions"],
            "own_castled": g[f"{color}_castled"],
            "own_castle_side": g[f"{color}_castle_side"],
            "own_first_capture_move": g[f"{color}_first_capture_move"],
            "own_early_queen_moves": g[f"{color}_early_queen_moves"],
            "own_territory_depth": g[f"{color}_territory_depth_max"],
            "first_own_move": g[f"first_{color}_move"],
        })
        for piece in PIECES:
            row[f"own_{piece}_moves"] = g[f"{color}_{piece}_moves"]

        # Optional board-aware features. These override SAN approximations where they are better.
        board_map = {
            "own_board_checks_given": f"checks_given_{color}",
            "own_board_check_density": f"check_density_{color}",
            "own_board_first_capture_move": f"first_capture_move_{color}",
            "own_board_castle_side": f"castle_side_{color}",
            "own_consec_same_piece": f"consec_same_piece_{color}",
            "own_legal_moves_move5": f"legal_moves_{color}_move5",
            "own_acpl": f"acpl_{color}",
            "own_blunder_density": f"blunder_density_{color}",
        }
        for out_col, in_col in board_map.items():
            if in_col in g:
                row[out_col] = g[in_col]
        terr_col = f"{color}_territory_depth"
        if terr_col in g:
            row["own_board_territory_depth"] = g[terr_col]

        clock_records = [_clock_game_features(r, color) for _, r in g.iterrows()]
        clock_df = pd.DataFrame(clock_records, index=row.index)
        row["own_mean_move_seconds"] = clock_df["mean_move_seconds"].values
        row["own_time_pressure_30s_rate"] = clock_df["time_pressure_30s_rate"].values
        rows.append(row)

    out = pd.concat(rows, ignore_index=True)
    out["own_castled"] = out["own_castled"].fillna(False).astype(bool)
    out["own_castle_kingside"] = (out["own_castle_side"] == "kingside").astype(int)
    out["own_castle_queenside"] = (out["own_castle_side"] == "queenside").astype(int)

    denom = pd.to_numeric(out["own_moves"], errors="coerce").replace(0, np.nan)
    out["capture_rate_game"] = out["own_captures"] / denom
    out["check_rate_game"] = out["own_checks"] / denom
    out["promotion_rate_game"] = out["own_promotions"] / denom
    out["queen_promotion_rate_game"] = out["own_queen_promotions"] / denom
    out["early_queen_move_rate_game"] = out["own_early_queen_moves"] / denom
    for piece in PIECES:
        out[f"{piece}_move_rate_game"] = out[f"own_{piece}_moves"] / denom
    if "own_consec_same_piece" in out:
        out["consec_same_piece_rate_game"] = out["own_consec_same_piece"] / denom
    return out


def _entropy(series: pd.Series) -> float:
    p = series.dropna().value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def parse_to_player_data(
    raw_df: pd.DataFrame,
    *,
    min_games: int = 20,
    filter_10_0_rapid: bool = True,
    include_board_features: bool = False,
    board_moves_col: str | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    df = filter_rated_standard_10_0_rapid(raw_df) if filter_10_0_rapid else raw_df.copy()
    pg = parse_to_player_game_data(df, include_board_features=include_board_features, board_moves_col=board_moves_col, n_jobs=n_jobs)

    agg = {
        "n_games": ("game_id", "count"),
        "elo_mean": ("elo", "mean"),
        "elo_median": ("elo", "median"),
        "elo_std": ("elo", "std"),
        "opponent_elo_mean": ("opponent_elo", "mean"),
        "opponent_elo_median": ("opponent_elo", "median"),
        "win_rate": ("result_score", "mean"),
        "avg_game_length": ("num_moves", "mean"),
        "capture_rate": ("capture_rate_game", "mean"),
        "check_rate": ("check_rate_game", "mean"),
        "castle_rate": ("own_castled", "mean"),
        "kingside_castle_rate": ("own_castle_kingside", "mean"),
        "queenside_castle_rate": ("own_castle_queenside", "mean"),
        "promotion_rate": ("promotion_rate_game", "mean"),
        "queen_promotion_rate": ("queen_promotion_rate_game", "mean"),
        "first_capture_move_mean": ("own_first_capture_move", "mean"),
        "early_queen_move_rate": ("early_queen_move_rate_game", "mean"),
        "territory_depth_mean": ("own_territory_depth", "mean"),
        "mean_move_seconds": ("own_mean_move_seconds", "mean"),
        "time_pressure_30s_rate": ("own_time_pressure_30s_rate", "mean"),
        "first_move_diversity": ("first_own_move", "nunique"),
        "opening_diversity": ("opening_san_6ply", "nunique"),
    }
    for piece in PIECES:
        agg[f"{piece}_move_rate"] = (f"{piece}_move_rate_game", "mean")
    optional = {
        "board_territory_depth_mean": ("own_board_territory_depth", "mean"),
        "consec_same_piece_rate": ("consec_same_piece_rate_game", "mean"),
        "legal_moves_move5_mean": ("own_legal_moves_move5", "mean"),
        "acpl_mean": ("own_acpl", "mean"),
        "blunder_density_mean": ("own_blunder_density", "mean"),
    }
    for key, spec in optional.items():
        if spec[0] in pg:
            agg[key] = spec

    out = pg.groupby("player").agg(**agg)
    out["elo_std"] = out["elo_std"].fillna(0.0)
    out["first_move_diversity_rate"] = out["first_move_diversity"] / out["n_games"]
    out["opening_diversity_rate"] = out["opening_diversity"] / out["n_games"]

    ent = pg.groupby("player").agg(
        first_move_entropy=("first_own_move", _entropy),
        opening_entropy=("opening_san_6ply", _entropy),
    )
    out = out.join(ent)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.query("n_games >= @min_games").reset_index()

    # Requested leftmost order: Elo columns first. Player is kept immediately after them.
    first_cols = ["elo_mean", "elo_median", "elo_std", "player", "n_games", "win_rate"]
    other_cols = [c for c in out.columns if c not in first_cols]
    return out[first_cols + other_cols]


def select_summary_features(player_df: pd.DataFrame, *, drop_missing: bool = False) -> pd.DataFrame:
    cols = [c for c in SUMMARY_FEATURES if c in player_df]
    if drop_missing:
        cols = [c for c in cols if player_df[c].notna().any()]
    prefix = [c for c in ["player", "elo_mean", "elo_median", "elo_std", "n_games", "win_rate"] if c in player_df]
    return player_df[prefix + cols].copy()


def feature_columns(player_df: pd.DataFrame, *, include_extra_numeric: bool = False) -> list[str]:
    if include_extra_numeric:
        numeric = player_df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in NON_TRAIN_COLUMNS]
    return [c for c in SUMMARY_FEATURES if c in player_df]


def drop_correlated_features(df: pd.DataFrame, columns: Sequence[str], *, threshold: float = 0.95) -> list[str]:
    corr = df[list(columns)].corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {col for col in upper.columns if any(upper[col] > threshold)}
    return [c for c in columns if c not in drop]
