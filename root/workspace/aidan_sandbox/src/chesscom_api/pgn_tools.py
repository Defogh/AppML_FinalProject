# Help with getting pgn files from chess.com

import re


LINK_RE = re.compile(r'\[Link "([^"]+)"\]')
SITE_RE = re.compile(r'\[Site "([^"]+)"\]')


# Turns multi-game pgn into individual games
def split_pgn_games(pgn_text: str) -> list[str]:
  games = []
  current_game = []

  for line in pgn_text.splitlines():
    if line.startswith("[Event ") and current_game:
      games.append("\n".join(current_game).strip())
      current_game = []

    current_game.append(line)

  if current_game:
    game = "\n".join(current_game).strip()

    if game:
      games.append(game)

  return games


# Gets the Chess.com game URL for deduplication
def extract_game_key(game_pgn: str) -> str | None:
  link_match = LINK_RE.search(game_pgn)

  if link_match is not None:
    return link_match.group(1)

  site_match = SITE_RE.search(game_pgn)

  if site_match is not None:
    return site_match.group(1)

  return None


# Keeps clean blank lines between games
def normalize_pgn_spacing(game_pgn: str) -> str:
  return game_pgn.strip() + "\n\n"
