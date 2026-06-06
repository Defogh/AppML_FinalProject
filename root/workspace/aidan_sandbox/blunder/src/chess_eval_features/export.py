"""Batch processing and export helpers."""

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from chess_eval_features.extract import build_sequence_tables
from chess_eval_features.features import (
  build_conservative_features,
  merge_features_with_game_metadata,
)
from chess_eval_features.parser import PgnZstParser


def stabilize_export_dtypes(df):
  """Force stable dtypes across batches before writing."""
  df = df.copy()

  float_cols = [
    "result_white",
    "avg_elo",
    "time_base_seconds",
    "time_increment_seconds",
    "eval_pawns",
    "mate_distance",
    "clock_seconds",
    "eval_proxy",
    "eval_delta_proxy",
    "abs_eval_change",
    "white_loss_proxy",
    "black_loss_proxy",
    "white_material",
    "black_material",
    "total_material",
    "white_non_pawn_material",
    "black_non_pawn_material",
    "total_non_pawn_material",
    "material_imbalance_white",
    "non_pawn_imbalance_white",
    "phase_progress",
    "opening_like_weight",
    "middlegame_like_weight",
    "endgame_like_weight",
  ]

  int_cols = [
    "game_index",
    "ply",
    "move_number",
    "white_elo",
    "black_elo",
    "elo_diff_white_minus_black",
    "white_queen_count",
    "black_queen_count",
    "white_piece_count",
    "black_piece_count",
  ]

  bool_cols = [
    "is_mate_eval",
    "has_numeric_eval",
    "has_any_eval",
    "board_parse_ok",
    "both_queens_present",
    "no_queens_present",
  ]

  for col in float_cols:
    if col in df.columns:
      df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
      ).astype("float64")

  for col in int_cols:
    if col in df.columns:
      df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
      ).astype("Int64")

  for col in bool_cols:
    if col in df.columns:
      df[col] = df[col].astype("boolean")

  return df


class TableWriter:
  def __init__(self, path, file_format):
    self.path = path
    self.file_format = file_format
    self._writer = None
    self._schema = None
    self._csv_header_written = False

  def write(self, df):
    if len(df) == 0:
      return

    df = stabilize_export_dtypes(df)

    if self.file_format == "csv":
      df.to_csv(
        self.path,
        mode="a",
        index=False,
        header=not self._csv_header_written,
      )
      self._csv_header_written = True
      return

    table = pa.Table.from_pandas(
      df,
      preserve_index=False,
    )

    if self._writer is None:
      self._schema = table.schema
      self._writer = pq.ParquetWriter(
        self.path,
        self._schema,
      )
    else:
      table = table.cast(self._schema)

    self._writer.write_table(table)

  def close(self):
    if self._writer is not None:
      self._writer.close()
      self._writer = None


def batched(iterable, batch_size):
  batch = []

  for item in iterable:
    batch.append(item)

    if len(batch) >= batch_size:
      yield batch
      batch = []

  if batch:
    yield batch


def make_feature_dictionary():
  rows = [
    {
      "table": "plies",
      "column": "eval_pawns",
      "meaning": "Engine eval from White's perspective in pawn units.",
      "raw_or_derived": "raw-parsed",
      "unit": "pawns",
    },
    {
      "table": "plies",
      "column": "eval_proxy",
      "meaning": "Finite mate-aware numeric eval used for summaries.",
      "raw_or_derived": "derived",
      "unit": "pawns proxy",
    },
    {
      "table": "plies",
      "column": "phase_progress",
      "meaning": "Fraction of starting non-pawn material removed.",
      "raw_or_derived": "derived",
      "unit": "unitless",
    },
    {
      "table": "features",
      "column": "mean_abs_eval_change",
      "meaning": "Mean absolute change in eval_proxy per ply.",
      "raw_or_derived": "derived",
      "unit": "pawns proxy",
    },
    {
      "table": "features",
      "column": "white_mean_loss_proxy",
      "meaning": "Mean eval loss after White moves.",
      "raw_or_derived": "derived",
      "unit": "pawns proxy",
    },
    {
      "table": "features",
      "column": "black_mean_loss_proxy",
      "meaning": "Mean eval loss after Black moves.",
      "raw_or_derived": "derived",
      "unit": "pawns proxy",
    },
  ]

  return pd.DataFrame(rows)


def write_manifest(output_dir, manifest):
  manifest_path = Path(output_dir) / "export_manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2))


def process_pgn_to_directory(
  input_path,
  output_dir,
  n_eval_games=None,
  batch_size=1000,
  file_format="parquet",
  mate_score=20.0,
  clip_value=20.0,
  include_raw_pgn=False,
):
  """Stream a PGN.zst file and export games, plies, and features."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  suffix = "csv" if file_format == "csv" else "parquet"
  games_writer = TableWriter(output_dir / f"games.{suffix}", file_format)
  plies_writer = TableWriter(output_dir / f"plies.{suffix}", file_format)
  features_writer = TableWriter(output_dir / f"features.{suffix}", file_format)

  parser = PgnZstParser(input_path)

  n_found = 0
  n_batches = 0
  max_index_seen = -1

  try:
    game_iter = parser.iter_eval_games(max_games=n_eval_games)

    for games in batched(game_iter, batch_size):
      n_batches += 1
      n_found += len(games)
      max_index_seen = max(max_index_seen, games[-1].game_index)

      df_plies, df_games = build_sequence_tables(
        games,
        include_raw_pgn=include_raw_pgn,
      )
      df_features, df_plies_feat = build_conservative_features(
        df_plies,
        mate_score=mate_score,
        clip_value=clip_value,
      )
      df_features = merge_features_with_game_metadata(
        df_features,
        df_games,
      )

      games_writer.write(df_games)
      plies_writer.write(df_plies_feat)
      features_writer.write(df_features)

      print(
        f"batch={n_batches} "
        f"eval_games_written={n_found} "
        f"last_game_index={max_index_seen}"
      )
  finally:
    games_writer.close()
    plies_writer.close()
    features_writer.close()

  feature_dictionary = make_feature_dictionary()
  feature_dictionary.to_csv(
    output_dir / "feature_dictionary.csv",
    index=False,
  )

  manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_file": str(input_path),
    "output_dir": str(output_dir),
    "n_eval_games_requested": n_eval_games,
    "n_eval_games_found": n_found,
    "n_total_games_scanned_at_least": max_index_seen + 1,
    "batch_size": batch_size,
    "file_format": file_format,
    "mate_score": mate_score,
    "clip_value": clip_value,
    "include_raw_pgn": include_raw_pgn,
  }
  write_manifest(output_dir, manifest)

  return manifest
