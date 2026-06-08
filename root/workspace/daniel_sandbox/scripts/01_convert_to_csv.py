from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from funcs.pgn_ingest import convert_pgn_to_full_and_sample_csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PGN_PATH = DATA_DIR / "lichess_db_standard_rated_2017-09.pgn"
FULL_CSV = DATA_DIR / "lichess_2017_09_full.csv"
TRAIN_CSV = DATA_DIR / "lichess_2017_09_train_300k.csv"

info = convert_pgn_to_full_and_sample_csv(
    PGN_PATH,
    FULL_CSV,
    TRAIN_CSV,
    sample_size=300_000,
    random_state=42,
    chunk_size=100_000,
    full_include_moves_pgn=False,
    sample_include_moves_pgn=True,
    show_progress=True,
)

print("\nDone:")
for key, value in info.items():
    print(f"  {key}: {value}")
