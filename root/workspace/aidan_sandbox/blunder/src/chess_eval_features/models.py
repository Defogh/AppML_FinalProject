"""Small data containers used by the feature extraction pipeline."""

from dataclasses import dataclass


@dataclass
class PgnGame:
  """One raw PGN game with parsed tags and unmodified movetext."""

  game_index: int
  tags: dict
  movetext: str
  raw_pgn: str
