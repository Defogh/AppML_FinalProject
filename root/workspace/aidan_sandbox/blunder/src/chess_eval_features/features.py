"""Derived feature engineering for extracted chess ply tables."""

import re

import chess
import numpy as np
import pandas as pd

PIECE_VALUES = {
  chess.PAWN: 1,
  chess.KNIGHT: 3,
  chess.BISHOP: 3,
  chess.ROOK: 5,
  chess.QUEEN: 9,
  chess.KING: 0,
}

STARTING_NON_PAWN_MATERIAL = 62
STARTING_TOTAL_MATERIAL = 78


def add_eval_proxy(df_plies, mate_score=20.0, clip_value=20.0):
  df = df_plies.copy()

  df["has_numeric_eval"] = df["eval_pawns"].notna()
  df["has_any_eval"] = df["has_numeric_eval"] | df["is_mate_eval"]

  df["eval_proxy"] = df["eval_pawns"].copy()
  df["eval_proxy"] = df["eval_proxy"].clip(
    lower=-clip_value,
    upper=clip_value,
  )

  mate_mask = df["is_mate_eval"].fillna(False)
  mate_sign = np.sign(df.loc[mate_mask, "mate_distance"].astype(float))
  df.loc[mate_mask, "eval_proxy"] = mate_score * mate_sign

  df["eval_kind"] = "missing"
  df.loc[df["has_numeric_eval"], "eval_kind"] = "numeric"
  df.loc[df["is_mate_eval"], "eval_kind"] = "mate"

  return df


def add_eval_deltas(df_plies_feat):
  df = df_plies_feat.copy()
  df = df.sort_values(["game_index", "ply"])

  df["eval_delta_proxy"] = (
    df.groupby("game_index")["eval_proxy"].diff()
  )
  df["abs_eval_change"] = df["eval_delta_proxy"].abs()

  df["white_loss_proxy"] = np.nan
  df["black_loss_proxy"] = np.nan

  white_mask = df["side"] == "white"
  black_mask = df["side"] == "black"

  df.loc[white_mask, "white_loss_proxy"] = (
    -df.loc[white_mask, "eval_delta_proxy"]
  ).clip(lower=0)

  df.loc[black_mask, "black_loss_proxy"] = (
    df.loc[black_mask, "eval_delta_proxy"]
  ).clip(lower=0)

  return df


def safe_mean(series):
  clean = series.dropna()
  return np.nan if len(clean) == 0 else clean.mean()


def safe_std(series):
  clean = series.dropna()
  return np.nan if len(clean) <= 1 else clean.std()


def safe_min(series):
  clean = series.dropna()
  return np.nan if len(clean) == 0 else clean.min()


def safe_max(series):
  clean = series.dropna()
  return np.nan if len(clean) == 0 else clean.max()


def safe_last(series):
  clean = series.dropna()
  return np.nan if len(clean) == 0 else clean.iloc[-1]


def safe_first(series):
  clean = series.dropna()
  return np.nan if len(clean) == 0 else clean.iloc[0]


def safe_frac(numerator, denominator):
  if denominator == 0:
    return np.nan

  return numerator / denominator


def summarize_one_game(group):
  group = group.sort_values("ply")

  n_plies = len(group)
  n_eval = int(group["has_any_eval"].sum())
  n_numeric = int(group["has_numeric_eval"].sum())
  n_mate = int(group["is_mate_eval"].sum())
  n_missing = n_plies - n_eval

  eval_proxy = group["eval_proxy"]
  abs_eval = eval_proxy.abs()
  abs_change = group["abs_eval_change"]

  min_eval = safe_min(eval_proxy)
  max_eval = safe_max(eval_proxy)

  if pd.isna(min_eval) or pd.isna(max_eval):
    eval_range = np.nan
  else:
    eval_range = max_eval - min_eval

  row = {
    "game_index": group["game_index"].iloc[0],
    "n_plies": n_plies,
    "n_eval_available": n_eval,
    "frac_eval_available": safe_frac(n_eval, n_plies),
    "n_missing_eval": n_missing,
    "frac_missing_eval": safe_frac(n_missing, n_plies),
    "n_numeric_eval": n_numeric,
    "n_mate_eval": n_mate,
    "frac_mate_eval": safe_frac(n_mate, n_plies),
    "has_mate_eval": n_mate > 0,
    "first_eval_proxy": safe_first(eval_proxy),
    "final_eval_proxy": safe_last(eval_proxy),
    "mean_eval_proxy": safe_mean(eval_proxy),
    "std_eval_proxy": safe_std(eval_proxy),
    "min_eval_proxy": min_eval,
    "max_eval_proxy": max_eval,
    "eval_range_proxy": eval_range,
    "mean_abs_eval_proxy": safe_mean(abs_eval),
    "final_abs_eval_proxy": abs(safe_last(eval_proxy)),
    "mean_abs_eval_change": safe_mean(abs_change),
    "std_abs_eval_change": safe_std(abs_change),
    "max_abs_eval_change": safe_max(abs_change),
  }

  for threshold in [1, 2, 4]:
    n_large = int((abs_change > threshold).sum())
    n_possible = int(abs_change.notna().sum())

    row[f"n_eval_swings_gt_{threshold}"] = n_large
    row[f"frac_eval_swings_gt_{threshold}"] = safe_frac(
      n_large,
      n_possible,
    )

  return row


def add_player_loss_features(row, group):
  white_loss = group["white_loss_proxy"]
  black_loss = group["black_loss_proxy"]

  row["white_mean_loss_proxy"] = safe_mean(white_loss)
  row["black_mean_loss_proxy"] = safe_mean(black_loss)
  row["white_std_loss_proxy"] = safe_std(white_loss)
  row["black_std_loss_proxy"] = safe_std(black_loss)
  row["white_max_loss_proxy"] = safe_max(white_loss)
  row["black_max_loss_proxy"] = safe_max(black_loss)
  row["white_total_loss_proxy"] = white_loss.sum(skipna=True)
  row["black_total_loss_proxy"] = black_loss.sum(skipna=True)

  for threshold in [1, 2, 4]:
    row[f"white_n_losses_gt_{threshold}"] = int(
      (white_loss > threshold).sum()
    )
    row[f"black_n_losses_gt_{threshold}"] = int(
      (black_loss > threshold).sum()
    )

  return row


def clean_san_for_board(san_raw):
  san = str(san_raw).strip()
  san = san.replace("0-0-0", "O-O-O")
  san = san.replace("0-0", "O-O")
  san = re.sub(r"[!?]+$", "", san)

  return san


def material_for_color(board, color, include_pawns=True):
  total = 0

  for piece_type, value in PIECE_VALUES.items():
    if piece_type == chess.KING:
      continue

    if piece_type == chess.PAWN and not include_pawns:
      continue

    total += value * len(board.pieces(piece_type, color))

  return total


def count_pieces_for_color(board, color):
  n_pieces = 0

  for piece_type in PIECE_VALUES:
    n_pieces += len(board.pieces(piece_type, color))

  return n_pieces


def queen_count_for_color(board, color):
  return len(board.pieces(chess.QUEEN, color))


def board_features_after_move(board):
  white_material = material_for_color(
    board,
    chess.WHITE,
    include_pawns=True,
  )
  black_material = material_for_color(
    board,
    chess.BLACK,
    include_pawns=True,
  )
  white_non_pawn = material_for_color(
    board,
    chess.WHITE,
    include_pawns=False,
  )
  black_non_pawn = material_for_color(
    board,
    chess.BLACK,
    include_pawns=False,
  )

  total_material = white_material + black_material
  total_non_pawn = white_non_pawn + black_non_pawn

  white_queens = queen_count_for_color(board, chess.WHITE)
  black_queens = queen_count_for_color(board, chess.BLACK)

  phase_progress = 1.0 - (
    total_non_pawn / STARTING_NON_PAWN_MATERIAL
  )
  phase_progress = np.clip(phase_progress, 0.0, 1.0)

  return {
    "white_material": white_material,
    "black_material": black_material,
    "total_material": total_material,
    "white_non_pawn_material": white_non_pawn,
    "black_non_pawn_material": black_non_pawn,
    "total_non_pawn_material": total_non_pawn,
    "material_imbalance_white": white_material - black_material,
    "non_pawn_imbalance_white": white_non_pawn - black_non_pawn,
    "phase_progress": phase_progress,
    "white_queen_count": white_queens,
    "black_queen_count": black_queens,
    "both_queens_present": white_queens > 0 and black_queens > 0,
    "no_queens_present": white_queens == 0 and black_queens == 0,
    "white_piece_count": count_pieces_for_color(board, chess.WHITE),
    "black_piece_count": count_pieces_for_color(board, chess.BLACK),
  }


def add_board_state_features(df_plies_feat):
  df = df_plies_feat.copy()
  df = df.sort_values(["game_index", "ply"])

  feature_rows = []

  for _, group in df.groupby("game_index", sort=False):
    board = chess.Board()

    for idx, row in group.iterrows():
      san_clean = clean_san_for_board(row["san_raw"])
      parse_ok = True

      try:
        move = board.parse_san(san_clean)
        board.push(move)
      except ValueError:
        parse_ok = False

      features = board_features_after_move(board)
      features["row_index"] = idx
      features["board_parse_ok"] = parse_ok
      feature_rows.append(features)

  df_board = pd.DataFrame(feature_rows).set_index("row_index")
  df = df.join(df_board, how="left")

  return df


def add_soft_phase_weights(df_plies_feat):
  df = df_plies_feat.copy()

  progress = df["phase_progress"].clip(0.0, 1.0)
  both_queens = df["both_queens_present"].fillna(False).astype(float)
  no_queens = df["no_queens_present"].fillna(False).astype(float)

  opening_like = ((0.30 - progress) / 0.30).clip(0.0, 1.0)
  opening_like = opening_like * both_queens

  endgame_by_material = ((progress - 0.45) / 0.35).clip(0.0, 1.0)
  endgame_by_queens = no_queens * (
    (progress - 0.25) / 0.35
  ).clip(0.0, 1.0)

  endgame_like = np.maximum(endgame_by_material, endgame_by_queens)
  middlegame_like = 1.0 - np.maximum(opening_like, endgame_like)
  middlegame_like = middlegame_like.clip(0.0, 1.0)

  df["opening_like_weight"] = opening_like
  df["middlegame_like_weight"] = middlegame_like
  df["endgame_like_weight"] = endgame_like

  return df


def safe_weighted_mean(values, weights):
  mask = values.notna() & weights.notna()
  clean_values = values[mask]
  clean_weights = weights[mask]
  weight_sum = clean_weights.sum()

  if weight_sum <= 0:
    return np.nan

  return (clean_values * clean_weights).sum() / weight_sum


def safe_weighted_sum(values, weights):
  mask = values.notna() & weights.notna()
  clean_values = values[mask]
  clean_weights = weights[mask]

  if len(clean_values) == 0:
    return np.nan

  return (clean_values * clean_weights).sum()


def add_phase_indicator_features(row, group):
  phase_names = [
    "opening_like",
    "middlegame_like",
    "endgame_like",
  ]

  for phase_name in phase_names:
    weight_col = f"{phase_name}_weight"
    weights = group[weight_col]

    row[f"{phase_name}_weight_sum"] = weights.sum()
    row[f"{phase_name}_weight_mean"] = safe_mean(weights)
    row[f"{phase_name}_mean_abs_eval"] = safe_weighted_mean(
      group["eval_proxy"].abs(),
      weights,
    )
    row[f"{phase_name}_mean_abs_eval_change"] = (
      safe_weighted_mean(group["abs_eval_change"], weights)
    )
    row[f"{phase_name}_white_mean_loss_proxy"] = (
      safe_weighted_mean(group["white_loss_proxy"], weights)
    )
    row[f"{phase_name}_black_mean_loss_proxy"] = (
      safe_weighted_mean(group["black_loss_proxy"], weights)
    )

  first_queenless = group.loc[
    group["no_queens_present"].fillna(False),
    "ply",
  ]
  first_simplified = group.loc[group["phase_progress"] >= 0.50, "ply"]

  row["final_phase_progress"] = safe_last(group["phase_progress"])
  row["mean_phase_progress"] = safe_mean(group["phase_progress"])
  row["max_phase_progress"] = safe_max(group["phase_progress"])
  row["final_total_non_pawn_material"] = safe_last(
    group["total_non_pawn_material"]
  )
  row["final_material_imbalance_white"] = safe_last(
    group["material_imbalance_white"]
  )
  row["first_queenless_ply"] = (
    first_queenless.min() if len(first_queenless) > 0 else np.nan
  )
  row["first_simplified_ply"] = (
    first_simplified.min() if len(first_simplified) > 0 else np.nan
  )

  return row


def build_conservative_features(
  df_plies,
  mate_score=20.0,
  clip_value=20.0,
):
  """Add per-ply features and aggregate game-level features."""
  df_work = add_eval_proxy(
    df_plies,
    mate_score=mate_score,
    clip_value=clip_value,
  )
  df_work = add_eval_deltas(df_work)
  df_work = add_board_state_features(df_work)
  df_work = add_soft_phase_weights(df_work)

  rows = []

  for _, group in df_work.groupby("game_index"):
    row = summarize_one_game(group)
    row = add_player_loss_features(row, group)
    row = add_phase_indicator_features(row, group)
    rows.append(row)

  df_features = pd.DataFrame(rows)
  df_features = df_features.sort_values("game_index")
  df_features = df_features.reset_index(drop=True)

  return df_features, df_work


def merge_features_with_game_metadata(df_features, df_games):
  metadata_cols = [
    "game_index",
    "result",
    "result_white",
    "white_elo",
    "black_elo",
    "avg_elo",
    "elo_diff_white_minus_black",
    "event",
    "eco",
    "opening",
    "time_control",
    "time_base_seconds",
    "time_increment_seconds",
    "termination",
  ]

  metadata_cols = [col for col in metadata_cols if col in df_games.columns]

  return df_features.merge(
    df_games[metadata_cols],
    on="game_index",
    how="left",
  )
