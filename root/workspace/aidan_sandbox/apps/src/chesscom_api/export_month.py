import argparse
import asyncio
from pathlib import Path

from chesscom_api.chess_client import ChessComClient
from chesscom_api.pgn_tools import (
  extract_game_key,
  normalize_pgn_spacing,
  split_pgn_games,
)


def read_usernames(path: Path) -> list[str]:
  usernames = []

  for line in path.read_text().splitlines():
    username = line.strip().lower()

    if not username:
      continue

    if username.startswith("#"):
      continue

    usernames.append(username)

  return usernames


async def export_month(
  users_file: Path,
  output_file: Path,
  year: int,
  month: int,
  delay_seconds: float,
) -> None:
  client = ChessComClient()
  usernames = read_usernames(users_file)

  seen_games: set[str] = set()
  users_ok = 0
  users_failed = 0
  games_seen = 0
  games_written = 0
  duplicates_skipped = 0

  output_file.parent.mkdir(parents=True, exist_ok=True)

  with output_file.open("w", encoding="utf-8") as out:
    for index, username in enumerate(usernames, start=1):
      print(f"[{index}/{len(usernames)}] Fetching {username}")

      try:
        pgn_text = await client.get_player_games_by_month_pgn(
          username,
          year,
          month,
        )
      except Exception as error:
        users_failed += 1
        print(f"  failed: {error}")
        await asyncio.sleep(delay_seconds)
        continue

      users_ok += 1
      games = split_pgn_games(pgn_text)
      print(f"  found {len(games)} games")

      for game in games:
        games_seen += 1
        game_key = extract_game_key(game)

        if game_key is not None and game_key in seen_games:
          duplicates_skipped += 1
          continue
        
        if game_key is not None:
          seen_games.add(game_key)

        out.write(normalize_pgn_spacing(game))
        games_written += 1

      await asyncio.sleep(delay_seconds)

  print()
  print("Export complete")
  print(f"Users succeeded:      {users_ok}")
  print(f"Users failed:         {users_failed}")
  print(f"Games seen:           {games_seen}")
  print(f"Games written:        {games_written}")
  print(f"Duplicates skipped:   {duplicates_skipped}")
  print(f"Output file:          {output_file}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Export Chess.com monthly PGN data.",
  )

  parser.add_argument(
    "--users-file",
    type=Path,
    required=True,
    help="Path to a text file containing one username per line.",
  )

  parser.add_argument(
    "--year",
    type=int,
    required=True,
    help="Year to export, for example 2024.",
  )

  parser.add_argument(
    "--month",
    type=int,
    required=True,
    help="Month to export, from 1 to 12.",
  )

  parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Output PGN file path.",
  )

  parser.add_argument(
    "--delay-seconds",
    type=float,
    default=1.0,
    help="Delay between Chess.com requests.",
  )

  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if args.month < 1 or args.month > 12:
    raise ValueError("Month must be between 1 and 12.")

  asyncio.run(
    export_month(
      users_file=args.users_file,
      output_file=args.output,
      year=args.year,
      month=args.month,
      delay_seconds=args.delay_seconds,
    )
  )


if __name__ == "__main__":
  main()