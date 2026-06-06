"""Command line interface for data processing."""

import argparse

from chess_eval_features.export import process_pgn_to_directory


def parse_args():
  parser = argparse.ArgumentParser(
    description="Extract eval-game features from a Lichess PGN.zst file.",
  )
  parser.add_argument(
    "--input",
    required=True,
    help="Path to .pgn.zst input file.",
  )
  parser.add_argument(
    "--output-dir",
    required=True,
    help="Directory where processed files are written.",
  )
  parser.add_argument(
    "--n-eval-games",
    type=int,
    default=None,
    help="Maximum number of eval games to process. Default: all.",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=1000,
    help="Number of eval games processed per batch.",
  )
  parser.add_argument(
    "--format",
    choices=["parquet", "csv"],
    default="parquet",
    help="Output table format.",
  )
  parser.add_argument(
    "--mate-score",
    type=float,
    default=20.0,
    help="Numeric proxy assigned to forced mate evals.",
  )
  parser.add_argument(
    "--clip-value",
    type=float,
    default=20.0,
    help="Clip ordinary evals to +/- this value.",
  )
  parser.add_argument(
    "--include-raw-pgn",
    action="store_true",
    help="Include raw PGN text in games table. Can be large.",
  )

  return parser.parse_args()


def main():
  args = parse_args()
  manifest = process_pgn_to_directory(
    input_path=args.input,
    output_dir=args.output_dir,
    n_eval_games=args.n_eval_games,
    batch_size=args.batch_size,
    file_format=args.format,
    mate_score=args.mate_score,
    clip_value=args.clip_value,
    include_raw_pgn=args.include_raw_pgn,
  )

  print("Done.")
  print(manifest)


if __name__ == "__main__":
  main()
