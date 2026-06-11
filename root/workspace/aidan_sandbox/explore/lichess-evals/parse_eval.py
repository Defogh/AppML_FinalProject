#!/usr/bin/env python3

"""
Filter a .pgn.zst file to games containing [%eval ...].

Outputs:
  - filtered .pgn.zst containing only games with evals
  - summary.json
  - summary.txt
  - plots/*.png

Usage:
  python3 filter_eval_pgn_zst.py input.pgn.zst

Optional:
  python3 filter_eval_pgn_zst.py input.pgn.zst \
    --output eval_games.pgn.zst \
    --summary-dir eval_summary
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import statistics
import urllib.request
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import zstandard as zstd


HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
EVAL_RE = re.compile(r"\[%eval\s+([^\]\s]+)\]")
CLK_RE = re.compile(r"\[%clk\s+([^\]]+)\]")
FULLMOVE_RE = re.compile(r"(?<!\d)(\d+)\.(?:\.\.)?")
LICHESS_STANDARD_RE = re.compile(
  r"^lichess_db_standard_rated_(\d{4})-(\d{2})\.pgn\.zst$"
)


def text_reader(path: Path) -> io.TextIOBase:
  raw = open(path, "rb")

  if path.suffix == ".zst":
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(raw)
    return io.TextIOWrapper(
      reader,
      encoding="utf-8",
      errors="replace",
    )

  return io.TextIOWrapper(
    raw,
    encoding="utf-8",
    errors="replace",
  )


def zst_writer(path: Path, level: int) -> io.TextIOBase:
  raw = open(path, "wb")
  cctx = zstd.ZstdCompressor(level=level)
  writer = cctx.stream_writer(raw)

  return io.TextIOWrapper(
    writer,
    encoding="utf-8",
  )


def iter_games(path: Path):
  current = []

  with text_reader(path) as f:
    for line in f:
      if line.startswith("[Event ") and current:
        yield "".join(current).strip() + "\n\n"
        current = [line]
      else:
        current.append(line)

      if current:
        yield "".join(current).strip() + "\n\n"


def ensure_lichess_pgn_zst(path: Path) -> Path:
  if path.exists():
    return path

  if LICHESS_STANDARD_RE.match(path.name) is None:
    raise FileNotFoundError(
      f"Missing PGN input: {path}\n"
      "Only canonical Lichess standard monthly files can be downloaded "
      "automatically. Expected a filename like "
      "lichess_db_standard_rated_2017-05.pgn.zst."
    )

  url = f"https://database.lichess.org/standard/{path.name}"
  path.parent.mkdir(parents=True, exist_ok=True)
  print(f"Downloading missing Lichess PGN: {url}")
  urllib.request.urlretrieve(url, path)
  print(f"Wrote: {path}")
  return path


def parse_headers(game: str) -> dict[str, str]:
  headers = {}

  for line in game.splitlines():
    if not line.startswith("["):
      break

    match = HEADER_RE.match(line)
    if match:
      key, value = match.groups()
      headers[key] = value

  return headers


def safe_int(value: Optional[str]) -> Optional[int]:
  if value is None:
    return None

  try:
    return int(value)
  except ValueError:
    return None


def speed_from_headers(headers: dict[str, str]) -> str:
  event = headers.get("Event", "").lower()
  tc = headers.get("TimeControl", "")

  for speed in [
    "bullet",
    "blitz",
    "rapid",
    "classical",
    "correspondence",
  ]:
    if speed in event:
      return speed.title()

  try:
    base_s, inc_s = tc.split("+")
    base = int(base_s)
    inc = int(inc_s)
    estimated = base + 40 * inc

    if estimated < 180:
      return "Bullet"
    if estimated < 480:
      return "Blitz"
    if estimated < 1500:
      return "Rapid"

    return "Classical"
  except Exception:
    return "Unknown"


def parse_eval(token: str) -> tuple[Optional[int], Optional[int]]:
  token = token.strip()

  if token.startswith("#"):
    try:
      return None, int(token[1:])
    except ValueError:
      return None, None

  try:
    return round(float(token) * 100), None
  except ValueError:
    return None, None


def game_info(game: str) -> dict:
  headers = parse_headers(game)

  eval_tokens = EVAL_RE.findall(game)
  clock_tokens = CLK_RE.findall(game)

  numeric = []
  mates = []

  for token in eval_tokens:
    cp, mate = parse_eval(token)

    if cp is not None:
      numeric.append(cp)
    elif mate is not None:
      mates.append(mate)

  move_nums = [
    int(x)
    for x in FULLMOVE_RE.findall(game)
  ]

  white_elo = safe_int(headers.get("WhiteElo"))
  black_elo = safe_int(headers.get("BlackElo"))

  avg_elo = None
  if white_elo is not None and black_elo is not None:
    avg_elo = (white_elo + black_elo) / 2

  return {
    "has_eval": bool(eval_tokens),
    "result": headers.get("Result", "Unknown"),
    "termination": headers.get("Termination", "Unknown"),
    "speed": speed_from_headers(headers),
    "time_control": headers.get("TimeControl", "Unknown"),
    "white_elo": white_elo,
    "black_elo": black_elo,
    "avg_elo": avg_elo,
    "fullmoves": max(move_nums) if move_nums else 0,
    "eval_count": len(eval_tokens),
    "numeric_eval_count": len(numeric),
    "mate_eval_count": len(mates),
    "clock_count": len(clock_tokens),
    "first_eval_cp": numeric[0] if numeric else None,
    "last_eval_cp": numeric[-1] if numeric else None,
  }


def empty_stats(name: str) -> dict:
  return {
    "name": name,
    "games": 0,
    "games_with_eval": 0,
    "games_with_clock": 0,
    "eval_tags": 0,
    "numeric_eval_tags": 0,
    "mate_eval_tags": 0,
    "clock_tags": 0,
    "fullmoves": [],
    "avg_elos": [],
    "first_eval_cps": [],
    "last_eval_cps": [],
    "results": collections.Counter(),
    "terminations": collections.Counter(),
    "speeds": collections.Counter(),
    "time_controls": collections.Counter(),
  }


def add_game(stats: dict, info: dict) -> None:
  stats["games"] += 1

  if info["has_eval"]:
    stats["games_with_eval"] += 1

  if info["clock_count"] > 0:
    stats["games_with_clock"] += 1

  stats["eval_tags"] += info["eval_count"]
  stats["numeric_eval_tags"] += info["numeric_eval_count"]
  stats["mate_eval_tags"] += info["mate_eval_count"]
  stats["clock_tags"] += info["clock_count"]

  if info["fullmoves"]:
    stats["fullmoves"].append(info["fullmoves"])

  if info["avg_elo"] is not None:
    stats["avg_elos"].append(info["avg_elo"])

  if info["first_eval_cp"] is not None:
    stats["first_eval_cps"].append(info["first_eval_cp"])

  if info["last_eval_cp"] is not None:
    stats["last_eval_cps"].append(info["last_eval_cp"])

  stats["results"][info["result"]] += 1
  stats["terminations"][info["termination"]] += 1
  stats["speeds"][info["speed"]] += 1
  stats["time_controls"][info["time_control"]] += 1


def summarize_nums(values: list[float]) -> dict:
  if not values:
    return {
      "count": 0,
      "mean": None,
      "median": None,
      "min": None,
      "max": None,
    }

  return {
    "count": len(values),
    "mean": round(statistics.mean(values), 3),
    "median": round(statistics.median(values), 3),
    "min": round(min(values), 3),
    "max": round(max(values), 3),
  }


def counter_dict(counter: collections.Counter, top_n=None) -> dict:
  return {
    str(k): int(v)
    for k, v in counter.most_common(top_n)
  }


def pct(part: int, whole: int) -> str:
  if whole == 0:
    return "0.00%"

  return f"{100 * part / whole:.2f}%"


def stats_json(stats: dict) -> dict:
  games = stats["games"]

  eval_pct = 0
  clock_pct = 0

  if games:
    eval_pct = round(100 * stats["games_with_eval"] / games, 4)
    clock_pct = round(100 * stats["games_with_clock"] / games, 4)

  return {
    "name": stats["name"],
    "games": games,
    "games_with_eval": stats["games_with_eval"],
    "games_with_clock": stats["games_with_clock"],
    "eval_tags": stats["eval_tags"],
    "numeric_eval_tags": stats["numeric_eval_tags"],
    "mate_eval_tags": stats["mate_eval_tags"],
    "clock_tags": stats["clock_tags"],
    "pct_games_with_eval": eval_pct,
    "pct_games_with_clock": clock_pct,
    "fullmoves": summarize_nums(stats["fullmoves"]),
    "avg_elo": summarize_nums(stats["avg_elos"]),
    "first_eval_cp": summarize_nums(stats["first_eval_cps"]),
    "last_eval_cp": summarize_nums(stats["last_eval_cps"]),
    "results": counter_dict(stats["results"]),
    "terminations": counter_dict(stats["terminations"]),
    "speeds": counter_dict(stats["speeds"]),
    "top_time_controls": counter_dict(
      stats["time_controls"],
      top_n=20,
    ),
  }


def write_summary_txt(
  path: Path,
  input_path: Path,
  output_path: Path,
  full: dict,
  eval_only: dict,
) -> None:
  full_j = stats_json(full)
  eval_j = stats_json(eval_only)

  lines = []

  lines.append("PGN eval-filter summary")
  lines.append("=" * 72)
  lines.append(f"Input file:  {input_path}")
  lines.append(f"Output file: {output_path}")
  lines.append("")

  lines.append("Full dataset")
  lines.append("-" * 72)
  lines.append(f"Games:             {full['games']:,}")
  lines.append(
    "Games with eval:   "
    f"{full['games_with_eval']:,} "
    f"({pct(full['games_with_eval'], full['games'])})"
  )
  lines.append(
    "Games with clocks: "
    f"{full['games_with_clock']:,} "
    f"({pct(full['games_with_clock'], full['games'])})"
  )
  lines.append(f"Eval tags:         {full['eval_tags']:,}")
  lines.append(f"Numeric eval tags: {full['numeric_eval_tags']:,}")
  lines.append(f"Mate eval tags:    {full['mate_eval_tags']:,}")
  lines.append(f"Clock tags:        {full['clock_tags']:,}")
  lines.append(f"Fullmoves:         {full_j['fullmoves']}")
  lines.append(f"Average Elo:       {full_j['avg_elo']}")
  lines.append(f"Speeds:            {full_j['speeds']}")
  lines.append(f"Results:           {full_j['results']}")
  lines.append(f"Terminations:      {full_j['terminations']}")
  lines.append("")

  lines.append("Filtered eval subset")
  lines.append("-" * 72)
  lines.append(f"Games written:     {eval_only['games']:,}")
  lines.append(f"Eval tags:         {eval_only['eval_tags']:,}")
  lines.append(
    f"Numeric eval tags: {eval_only['numeric_eval_tags']:,}"
  )
  lines.append(f"Mate eval tags:    {eval_only['mate_eval_tags']:,}")
  lines.append(f"Clock tags:        {eval_only['clock_tags']:,}")
  lines.append(f"Fullmoves:         {eval_j['fullmoves']}")
  lines.append(f"Average Elo:       {eval_j['avg_elo']}")
  lines.append(f"First eval cp:     {eval_j['first_eval_cp']}")
  lines.append(f"Last eval cp:      {eval_j['last_eval_cp']}")
  lines.append(f"Speeds:            {eval_j['speeds']}")
  lines.append(f"Results:           {eval_j['results']}")
  lines.append(f"Terminations:      {eval_j['terminations']}")
  lines.append("")

  lines.append("Notes")
  lines.append("-" * 72)
  lines.append("[%eval 0.35] means +35 centipawns for White.")
  lines.append("[%eval -1.20] means -120 centipawns, favoring Black.")
  lines.append("[%eval #5] means forced mate in 5.")
  lines.append("This script does not run Stockfish.")
  lines.append("It only extracts evals already present in the PGN.")

  path.write_text(
    "\n".join(lines),
    encoding="utf-8",
  )


def plot_bar(
  counter: collections.Counter,
  title: str,
  output_path: Path,
  top_n: int = 15,
) -> None:
  if not counter:
    return

  items = counter.most_common(top_n)
  labels = [str(k) for k, _ in items]
  values = [v for _, v in items]

  plt.figure(figsize=(10, 6))
  plt.bar(labels, values)
  plt.title(title)
  plt.ylabel("Games")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def plot_hist(
  values: list[float],
  title: str,
  xlabel: str,
  output_path: Path,
  bins: int = 50,
) -> None:
  if not values:
    return

  plt.figure(figsize=(10, 6))
  plt.hist(values, bins=bins)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel("Games")
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def plot_compare(
  labels: list[str],
  full_values: list[int],
  eval_values: list[int],
  title: str,
  output_path: Path,
) -> None:
  x_vals = list(range(len(labels)))
  width = 0.4

  left_x = [
    x - width / 2
    for x in x_vals
  ]

  right_x = [
    x + width / 2
    for x in x_vals
  ]

  plt.figure(figsize=(10, 6))
  plt.bar(left_x, full_values, width=width, label="Full dataset")
  plt.bar(right_x, eval_values, width=width, label="Eval subset")
  plt.title(title)
  plt.ylabel("Games")
  plt.xticks(x_vals, labels, rotation=45, ha="right")
  plt.legend()
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def make_plots(full: dict, eval_only: dict, plots_dir: Path) -> None:
  plots_dir.mkdir(parents=True, exist_ok=True)

  plot_hist(
    full["fullmoves"],
    "Game length distribution: full dataset",
    "Fullmoves",
    plots_dir / "full_dataset_game_lengths.png",
  )

  plot_hist(
    eval_only["fullmoves"],
    "Game length distribution: eval subset",
    "Fullmoves",
    plots_dir / "eval_subset_game_lengths.png",
  )

  plot_hist(
    full["avg_elos"],
    "Average Elo distribution: full dataset",
    "Average Elo",
    plots_dir / "full_dataset_avg_elo.png",
  )

  plot_hist(
    eval_only["avg_elos"],
    "Average Elo distribution: eval subset",
    "Average Elo",
    plots_dir / "eval_subset_avg_elo.png",
  )

  plot_hist(
    eval_only["first_eval_cps"],
    "First numeric eval in eval games",
    "Centipawns from White perspective",
    plots_dir / "eval_subset_first_eval_cp.png",
  )

  plot_hist(
    eval_only["last_eval_cps"],
    "Last numeric eval in eval games",
    "Centipawns from White perspective",
    plots_dir / "eval_subset_last_eval_cp.png",
  )

  plot_bar(
    full["speeds"],
    "Speed categories: full dataset",
    plots_dir / "full_dataset_speeds.png",
  )

  plot_bar(
    eval_only["speeds"],
    "Speed categories: eval subset",
    plots_dir / "eval_subset_speeds.png",
  )

  plot_bar(
    full["terminations"],
    "Terminations: full dataset",
    plots_dir / "full_dataset_terminations.png",
  )

  plot_bar(
    eval_only["terminations"],
    "Terminations: eval subset",
    plots_dir / "eval_subset_terminations.png",
  )

  speed_labels = sorted(
    set(full["speeds"].keys())
    | set(eval_only["speeds"].keys())
  )

  plot_compare(
    labels=speed_labels,
    full_values=[
      full["speeds"][label]
      for label in speed_labels
    ],
    eval_values=[
      eval_only["speeds"][label]
      for label in speed_labels
    ],
    title="Speed category comparison",
    output_path=plots_dir / "speed_comparison.png",
  )

  result_labels = sorted(
    set(full["results"].keys())
    | set(eval_only["results"].keys())
  )

  plot_compare(
    labels=result_labels,
    full_values=[
      full["results"][label]
      for label in result_labels
    ],
    eval_values=[
      eval_only["results"][label]
      for label in result_labels
    ],
    title="Result comparison",
    output_path=plots_dir / "result_comparison.png",
  )


def default_output(input_path: Path) -> Path:
  name = input_path.name

  if name.endswith(".pgn.zst"):
    new_name = name.replace(
      ".pgn.zst",
      ".with_eval.pgn.zst",
    )
    return input_path.with_name(new_name)

  if name.endswith(".zst"):
    new_name = name.replace(
      ".zst",
      ".with_eval.pgn.zst",
    )
    return input_path.with_name(new_name)

  return input_path.with_suffix(".with_eval.pgn.zst")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Extract PGN games containing [%eval ...].",
  )

  parser.add_argument(
    "input",
    type=Path,
    help="Input .pgn.zst or plain .pgn file",
  )

  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output .pgn.zst file",
  )

  parser.add_argument(
    "--summary-dir",
    type=Path,
    default=Path("eval_summary"),
    help="Directory for summary files and plots",
  )

  parser.add_argument(
    "--compression-level",
    type=int,
    default=10,
    help="Zstandard compression level",
  )

  parser.add_argument(
    "--progress-every",
    type=int,
    default=100_000,
    help="Print progress every N games",
  )

  return parser.parse_args()


def main() -> None:
  args = parse_args()

  input_path = ensure_lichess_pgn_zst(args.input)
  output_path = args.output or default_output(input_path)
  summary_dir = args.summary_dir
  plots_dir = summary_dir / "plots"

  summary_dir.mkdir(parents=True, exist_ok=True)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  full = empty_stats("full_dataset")
  eval_only = empty_stats("eval_subset")

  with zst_writer(output_path, args.compression_level) as out:
    for i, game in enumerate(iter_games(input_path), start=1):
      info = game_info(game)
      add_game(full, info)

      if info["has_eval"]:
        add_game(eval_only, info)
        out.write(game)

        if not game.endswith("\n\n"):
          out.write("\n\n")

      if args.progress_every and i % args.progress_every == 0:
        msg = (
          f"Processed {i:,} games | "
          f"eval games: {eval_only['games']:,} "
          f"({pct(eval_only['games'], full['games'])})"
        )
        print(msg)

  summary = {
    "input_file": str(input_path),
    "output_file": str(output_path),
    "full_dataset": stats_json(full),
    "eval_subset": stats_json(eval_only),
  }

  summary_json = summary_dir / "summary.json"
  summary_txt = summary_dir / "summary.txt"

  summary_json.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
  )

  write_summary_txt(
    path=summary_txt,
    input_path=input_path,
    output_path=output_path,
    full=full,
    eval_only=eval_only,
  )

  make_plots(full, eval_only, plots_dir)

  print("")
  print("Done.")
  print(f"Input games:      {full['games']:,}")
  print(
    "Eval games:       "
    f"{eval_only['games']:,} "
    f"({pct(eval_only['games'], full['games'])})"
  )
  print(f"Output PGN:       {output_path}")
  print(f"Summary JSON:     {summary_json}")
  print(f"Summary text:     {summary_txt}")
  print(f"Plots directory:  {plots_dir}")


if __name__ == "__main__":
  main()
