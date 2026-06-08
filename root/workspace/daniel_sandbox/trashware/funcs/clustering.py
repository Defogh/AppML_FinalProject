import numpy as np
import pandas as pd


def format_table(input_df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """
    Reformats raw lichess data into one of three useful table formats.

    Parameters
    ----------
    input_df:
        Input DataFrame containing parsed chess game data.

    table_type:
        "type1" -> one row per game
        "type2" -> two rows per game, one per player
        "type3" -> one row per player
    """
    table_type = table_type.lower().strip()

    if table_type not in {"type1", "type2", "type3"}:
        raise ValueError(f"Invalid table_type: {table_type}. Must be 'type1', 'type2', or 'type3'.")

    def require(df: pd.DataFrame, columns: list[str], name: str) -> None:
        missing = [c for c in columns if c not in df]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    def fill_missing(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
        for col, value in defaults.items():
            if col not in df:
                df[col] = value
        return df

    def safe_divide(a, b):
        return np.where(pd.to_numeric(b, errors="coerce") == 0, np.nan, a / b)

    def entropy(series: pd.Series) -> float:
        p = series.dropna().value_counts(normalize=True)
        return float(-(p * np.log2(p)).sum()) if len(p) else 0.0

    def result_to_white_score(df: pd.DataFrame) -> pd.Series:
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

        raise ValueError("Could not infer result_white_score. Need either 'result', 'winner', or 'result_white_score'.")

    # -------------------------------------------------------------------------
    # Type 1: one row per game
    # -------------------------------------------------------------------------
    output_df = input_df.copy().reset_index(drop=True)

    if "game_id" not in output_df:
        output_df.insert(0, "game_id", np.arange(len(output_df), dtype=int))

    require(output_df, ["white", "black", "white_elo", "black_elo"], "Type 1 table")

    if "result_white_score" not in output_df:
        output_df["result_white_score"] = result_to_white_score(output_df)

    if "num_halfmoves" not in output_df:
        move_col = (
            "num_moves" if "num_moves" in output_df
            else "num_fullmoves" if "num_fullmoves" in output_df
            else None
        )

        if move_col is None:
            raise ValueError("Type 1 table needs num_halfmoves, num_moves, or num_fullmoves.")

        output_df["num_halfmoves"] = (
            pd.to_numeric(output_df[move_col], errors="coerce")
            .mul(2)
            .round()
        )

    output_df["num_halfmoves"] = (
        pd.to_numeric(output_df["num_halfmoves"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "num_white_moves" not in output_df:
        output_df["num_white_moves"] = ((output_df["num_halfmoves"] + 1) // 2).astype(int)

    if "num_black_moves" not in output_df:
        output_df["num_black_moves"] = (output_df["num_halfmoves"] // 2).astype(int)

    if "num_fullmoves" not in output_df:
        output_df["num_fullmoves"] = output_df["num_halfmoves"] / 2

    if "num_moves" not in output_df:
        output_df["num_moves"] = output_df["num_halfmoves"] / 2

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

    if table_type == "type1":
        return output_df

    # -------------------------------------------------------------------------
    # Type 2: two rows per game, one from each player's perspective
    # -------------------------------------------------------------------------
    common_cols = [
        "game_id", "event", "site", "date", "time", "speed",
        "initial_seconds", "increment_seconds", "termination", "eco",
        "eco_family", "opening", "opening_san_4ply", "opening_san_6ply",
        "opening_san_10ply", "num_halfmoves",
    ]

    common = {c: output_df[c] for c in common_cols if c in output_df}

    player_rows = []

    for own, opp in [("white", "black"), ("black", "white")]:
        player_rows.append(pd.DataFrame({
            **common,
            "player": output_df[own],
            "opponent": output_df[opp],
            "color": own,
            "is_white": int(own == "white"),
            "elo": output_df[f"{own}_elo"],
            "opponent_elo": output_df[f"{opp}_elo"],
            "elo_diff_vs_opponent": output_df[f"{own}_elo"] - output_df[f"{opp}_elo"],
            "result_score": output_df["result_white_score"] if own == "white" else 1.0 - output_df["result_white_score"],
            "rating_diff": output_df[f"{own}_rating_diff"],
            "own_moves": output_df[f"num_{own}_moves"],
            "opp_moves": output_df[f"num_{opp}_moves"],
            "own_captures": output_df[f"{own}_captures"],
            "opp_captures": output_df[f"{opp}_captures"],
            "own_checks": output_df[f"{own}_checks"],
            "opp_checks": output_df[f"{opp}_checks"],
            "own_checkmates": output_df[f"{own}_checkmates"],
            "opp_checkmates": output_df[f"{opp}_checkmates"],
            "own_castled": output_df[f"{own}_castled"],
            "opp_castled": output_df[f"{opp}_castled"],
            "own_castle_side": output_df[f"{own}_castle_side"],
            "opp_castle_side": output_df[f"{opp}_castle_side"],
            "first_own_move": output_df[f"first_{own}_move"],
            "first_opp_move": output_df[f"first_{opp}_move"],
        }))

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

    if table_type == "type2":
        return output_df

    # -------------------------------------------------------------------------
    # Type 3: one row per player
    # -------------------------------------------------------------------------
    min_games = 20
    include_speed_distribution = True
    include_eco_distribution = True

    player_df = output_df

    output_df = player_df.groupby("player").agg(
        n_games=("game_id", "count"),
        median_elo=("elo", "median"),
        mean_elo=("elo", "mean"),
        elo_std=("elo", "std"),
        median_opponent_elo=("opponent_elo", "median"),
        mean_opponent_elo=("opponent_elo", "mean"),
        mean_result=("result_score", "mean"),
        white_rate=("is_white", "mean"),
        mean_game_length=("game_length_moves", "mean"),
        median_game_length=("game_length_moves", "median"),
        mean_initial_seconds=("initial_seconds", "mean"),
        mean_increment_seconds=("increment_seconds", "mean"),
        capture_rate=("own_capture_rate", "mean"),
        opponent_capture_rate=("opp_capture_rate", "mean"),
        check_rate=("own_check_rate", "mean"),
        opponent_check_rate=("opp_check_rate", "mean"),
        castle_rate=("own_castled", "mean"),
        kingside_castle_rate=("own_castle_kingside", "mean"),
        queenside_castle_rate=("own_castle_queenside", "mean"),
        first_move_diversity=("first_own_move", "nunique"),
        eco_diversity=("eco", "nunique"),
        opening_diversity=("opening", "nunique"),
    )

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

    if include_speed_distribution:
        speed_df = pd.crosstab(
            player_df["player"],
            player_df["speed"],
            normalize="index",
        ).add_prefix("speed_frac_")

        output_df = output_df.join(speed_df, how="left")

    if include_eco_distribution:
        eco_df = pd.crosstab(
            player_df["player"],
            player_df["eco_family"],
            normalize="index",
        ).add_prefix("eco_family_frac_")

        output_df = output_df.join(eco_df, how="left")

    output_df = (
        output_df
        .fillna(0.0)
        .query("n_games >= @min_games")
        .reset_index()
    )

    return output_df


