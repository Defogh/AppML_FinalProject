from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from funcs.player_features import parse_to_player_data, select_summary_features


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA_DIR / "lichess_2017_09_full.csv"
PLAYER_CSV = DATA_DIR / "players_10_0_full_summary_min10.csv"
SUMMARY_CSV = DATA_DIR / "players_10_0_full_summary_features_min10.csv"

MIN_GAMES = 10

print(f"reading {TRAIN_CSV}", flush=True)
raw = pd.read_csv(TRAIN_CSV)
print(f"loaded {len(raw):,} sampled 10+0 games", flush=True)

print(f"building one-row-per-player table, min_games={MIN_GAMES}", flush=True)
players = parse_to_player_data(
    raw,
    min_games=MIN_GAMES,
    filter_10_0_rapid=True,
    include_board_features=False,
)
summary = select_summary_features(players)

PLAYER_CSV.parent.mkdir(parents=True, exist_ok=True)
players.to_csv(PLAYER_CSV, index=False)
summary.to_csv(SUMMARY_CSV, index=False)

print("\nDone:")
print(f"  sampled_games: {len(raw):,}")
print(f"  players: {len(players):,}")
print(f"  player_csv: {PLAYER_CSV}")
print(f"  summary_csv: {SUMMARY_CSV}")
