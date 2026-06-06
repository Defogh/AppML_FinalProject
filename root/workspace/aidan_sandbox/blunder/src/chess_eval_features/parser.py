"""Streaming parser for compressed Lichess PGN files."""

from pathlib import Path
import io
import re

import zstandard as zstd

from chess_eval_features.models import PgnGame


class PgnZstParser:
  """Stream games from a ``.pgn.zst`` file without loading it fully."""

  def __init__(self, path):
    self.path = Path(path)

  def parse_first_n_with_eval(self, n_games):
    """Return the first ``n_games`` that contain ``%eval``."""
    eval_games = []
    n_scanned = 0

    for game in self.iter_eval_games():
      eval_games.append(game)
      n_scanned = game.game_index + 1

      if len(eval_games) >= n_games:
        break

    scan_report = {
      "n_requested_eval_games": n_games,
      "n_eval_games_found": len(eval_games),
      "n_total_games_scanned": n_scanned,
    }

    return eval_games, scan_report

  def iter_eval_games(self, max_games=None):
    """Yield games containing ``%eval``.

    Non-eval games are scanned but not converted into ``PgnGame`` objects.
    """
    n_found = 0

    for game_index, game_lines in enumerate(self._iter_raw_game_lines()):
      raw_pgn = "\n".join(game_lines).strip()

      if "%eval" not in raw_pgn:
        continue

      yield self._build_game(
        game_index=game_index,
        game_lines=game_lines,
      )

      n_found += 1

      if max_games is not None and n_found >= max_games:
        break

  def _iter_raw_game_lines(self):
    game_lines = []
    seen_movetext = False

    with open(self.path, "rb") as file:
      dctx = zstd.ZstdDecompressor()

      with dctx.stream_reader(file) as byte_stream:
        text_stream = io.TextIOWrapper(
          byte_stream,
          encoding="utf-8",
          errors="replace",
        )

        for line in text_stream:
          line = line.rstrip("\n")

          if self._is_new_game_start(
            line=line,
            game_lines=game_lines,
            seen_movetext=seen_movetext,
          ):
            yield game_lines
            game_lines = []
            seen_movetext = False

          game_lines.append(line)

          if self._is_movetext_line(line):
            seen_movetext = True

        if game_lines:
          yield game_lines

  def _is_new_game_start(self, line, game_lines, seen_movetext):
    if not game_lines:
      return False

    if not seen_movetext:
      return False

    return line.startswith("[Event ")

  def _is_movetext_line(self, line):
    if not line.strip():
      return False

    if line.startswith("[") and line.endswith("]"):
      return False

    return True

  def _build_game(self, game_index, game_lines):
    raw_pgn = "\n".join(game_lines).strip()
    tags = {}
    movetext_lines = []
    in_movetext = False

    for line in game_lines:
      if self._is_tag_line(line) and not in_movetext:
        key, value = self._parse_tag_line(line)
        tags[key] = value
        continue

      if line.strip():
        in_movetext = True

      if in_movetext:
        movetext_lines.append(line)

    movetext = "\n".join(movetext_lines).strip()

    return PgnGame(
      game_index=game_index,
      tags=tags,
      movetext=movetext,
      raw_pgn=raw_pgn,
    )

  def _is_tag_line(self, line):
    return bool(re.match(r'^\[[A-Za-z0-9_]+ ".*"\]$', line))

  def _parse_tag_line(self, line):
    match = re.match(r'^\[([A-Za-z0-9_]+) "(.*)"\]$', line)

    if match is None:
      raise ValueError(f"Could not parse tag line: {line}")

    return match.group(1), match.group(2)
