"""Raw sequence extraction from parsed PGN games."""

import re

import pandas as pd

TOKEN_PATTERN = re.compile(r"\{[^}]*\}|\S+")
EVAL_PATTERN = re.compile(r"\[%eval\s+([^\]]+)\]")
CLOCK_PATTERN = re.compile(r"\[%clk\s+([^\]]+)\]")

MOVE_NUMBER_PATTERN = re.compile(r"^\d+\.(?:\.\.)?$")
NAG_PATTERN = re.compile(r"^\$\d+$")

RESULT_TOKENS = {
  "1-0",
  "0-1",
  "1/2-1/2",
  "*",
}


def parse_eval_raw(eval_raw):
  """Parse a raw Lichess eval string.

  Numeric evals are returned in pawn units. Forced mate evals are kept
  separate because ``#3`` is not equivalent to ``+3`` pawns.
  """
  if eval_raw is None:
    return {
      "eval_pawns": None,
      "is_mate_eval": False,
      "mate_distance": None,
    }

  value = eval_raw.strip()

  if value.startswith("#"):
    mate_text = value.replace("#", "")

    return {
      "eval_pawns": None,
      "is_mate_eval": True,
      "mate_distance": int(mate_text),
    }

  return {
    "eval_pawns": float(value),
    "is_mate_eval": False,
    "mate_distance": None,
  }


def parse_clock_seconds(clock_raw):
  if clock_raw is None:
    return None

  pieces = clock_raw.strip().split(":")

  if len(pieces) == 3:
    hours = int(pieces[0])
    minutes = int(pieces[1])
    seconds = float(pieces[2])

    return 3600 * hours + 60 * minutes + seconds

  if len(pieces) == 2:
    minutes = int(pieces[0])
    seconds = float(pieces[1])

    return 60 * minutes + seconds

  return float(clock_raw)


def is_comment_token(token):
  return token.startswith("{") and token.endswith("}")


def is_move_number_token(token):
  return bool(MOVE_NUMBER_PATTERN.match(token))


def is_result_token(token):
  return token in RESULT_TOKENS


def is_nag_token(token):
  return bool(NAG_PATTERN.match(token))


def is_skip_token(token):
  return (
    is_move_number_token(token)
    or is_result_token(token)
    or is_nag_token(token)
  )


def extract_ply_rows_from_game(game):
  """Extract one row per ply from a parsed PGN game."""
  rows = []
  current_row_idx = None
  ply = 0

  tokens = TOKEN_PATTERN.findall(game.movetext)

  for token in tokens:
    if is_comment_token(token):
      if current_row_idx is None:
        continue

      comment_raw = token
      eval_matches = EVAL_PATTERN.findall(comment_raw)
      clock_matches = CLOCK_PATTERN.findall(comment_raw)

      eval_raw = eval_matches[0] if eval_matches else None
      clock_raw = clock_matches[0] if clock_matches else None

      eval_parts = parse_eval_raw(eval_raw)
      clock_seconds = parse_clock_seconds(clock_raw)

      rows[current_row_idx]["comment_raw"] = comment_raw
      rows[current_row_idx]["eval_raw"] = eval_raw
      rows[current_row_idx]["clock_raw"] = clock_raw
      rows[current_row_idx]["clock_seconds"] = clock_seconds
      rows[current_row_idx].update(eval_parts)

      continue

    if is_skip_token(token):
      continue

    ply += 1

    side = "white" if ply % 2 == 1 else "black"
    move_number = (ply + 1) // 2

    row = {
      "game_index": game.game_index,
      "ply": ply,
      "move_number": move_number,
      "side": side,
      "san_raw": token,
      "comment_raw": None,
      "eval_raw": None,
      "eval_pawns": None,
      "is_mate_eval": False,
      "mate_distance": None,
      "clock_raw": None,
      "clock_seconds": None,
    }

    rows.append(row)
    current_row_idx = len(rows) - 1

  return rows


def safe_int(value):
  if value is None:
    return None

  try:
    return int(value)
  except ValueError:
    return None


def parse_result_for_white(result):
  if result == "1-0":
    return 1.0

  if result == "0-1":
    return 0.0

  if result == "1/2-1/2":
    return 0.5

  return None


def parse_time_control(time_control):
  if time_control is None:
    return {
      "time_base_seconds": None,
      "time_increment_seconds": None,
    }

  if "+" not in time_control:
    return {
      "time_base_seconds": None,
      "time_increment_seconds": None,
    }

  base, increment = time_control.split("+", 1)

  return {
    "time_base_seconds": safe_int(base),
    "time_increment_seconds": safe_int(increment),
  }


def extract_game_metadata(game):
  tags = game.tags
  time_parts = parse_time_control(tags.get("TimeControl"))

  white_elo = safe_int(tags.get("WhiteElo"))
  black_elo = safe_int(tags.get("BlackElo"))

  if white_elo is None or black_elo is None:
    avg_elo = None
    elo_diff_white_minus_black = None
  else:
    avg_elo = 0.5 * (white_elo + black_elo)
    elo_diff_white_minus_black = white_elo - black_elo

  row = {
    "game_index": game.game_index,
    "event": tags.get("Event"),
    "site": tags.get("Site"),
    "white": tags.get("White"),
    "black": tags.get("Black"),
    "result": tags.get("Result"),
    "result_white": parse_result_for_white(tags.get("Result")),
    "white_elo": white_elo,
    "black_elo": black_elo,
    "avg_elo": avg_elo,
    "elo_diff_white_minus_black": elo_diff_white_minus_black,
    "eco": tags.get("ECO"),
    "opening": tags.get("Opening"),
    "time_control": tags.get("TimeControl"),
    "termination": tags.get("Termination"),
    "raw_pgn": game.raw_pgn,
  }

  row.update(time_parts)

  return row


def build_sequence_tables(games, include_raw_pgn=True):
  """Build one-row-per-ply and one-row-per-game tables."""
  ply_rows = []
  game_rows = []

  for game in games:
    ply_rows.extend(extract_ply_rows_from_game(game))
    game_row = extract_game_metadata(game)

    if not include_raw_pgn:
      game_row.pop("raw_pgn", None)

    game_rows.append(game_row)

  return pd.DataFrame(ply_rows), pd.DataFrame(game_rows)
