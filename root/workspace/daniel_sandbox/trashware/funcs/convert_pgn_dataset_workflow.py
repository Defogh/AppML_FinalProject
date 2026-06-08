"""
Convert one Lichess PGN/PGN.ZST file into project CSVs.

Outputs:
    1. Full raw CSV: compact, one row per game.
    2. 300K training CSV: one row per game, includes moves_pgn for optional
       board-aware feature extraction.

Run:
    python convert_pgn_dataset_workflow.py
"""

from pathlib import Path

from pgn_to_df_workflow import convert_pgn_to_full_and_sample_csv


DIR = Path.cwd().parents[1] / "data" / "daniel_data"

# Change to .pgn.zst if you use the raw Lichess download.
# .zst support requires: pip install zstandard
INPUT_PATH = DIR / "lichess_db_standard_rated_2014-07.pgn"

FULL_OUTPUT_PATH = DIR / "lichess_2014-07_raw_full.csv"
TRAIN_OUTPUT_PATH = DIR / "lichess_2014-07_raw_train_300k.csv"

info = convert_pgn_to_full_and_sample_csv(
    INPUT_PATH,
    FULL_OUTPUT_PATH,
    TRAIN_OUTPUT_PATH,
    sample_size=300_000,
    random_state=42,
    chunk_size=100_000,

    # Compact full file.
    full_include_moves_san=True,
    full_include_annotation_series=True,
    full_include_moves_pgn=False,

    # Richer 300K training sample, used for board-aware features and SAN AE.
    sample_include_moves_san=True,
    sample_include_annotation_series=True,
    sample_include_moves_pgn=True,
)

print(info)
