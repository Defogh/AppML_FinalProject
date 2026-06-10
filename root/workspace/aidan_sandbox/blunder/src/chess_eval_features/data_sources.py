"""Helpers for local data files used by the feature pipeline."""

from __future__ import annotations

from pathlib import Path
import re
import urllib.request


LICHESS_STANDARD_RE = re.compile(
  r"^lichess_db_standard_rated_(\d{4})-(\d{2})\.pgn\.zst$"
)


def lichess_standard_url(path: str | Path) -> str | None:
  """Return the Lichess download URL for a standard monthly PGN file."""
  name = Path(path).name

  if LICHESS_STANDARD_RE.match(name) is None:
    return None

  return f"https://database.lichess.org/standard/{name}"


def ensure_lichess_pgn_zst(path: str | Path) -> Path:
  """Download a known Lichess monthly PGN.zst file when it is missing."""
  path = Path(path)

  if path.exists():
    return path

  url = lichess_standard_url(path)
  if url is None:
    raise FileNotFoundError(
      f"Missing raw PGN file: {path}\n"
      "Only canonical Lichess standard monthly files can be downloaded "
      "automatically. Expected a filename like "
      "lichess_db_standard_rated_2017-05.pgn.zst."
    )

  path.parent.mkdir(parents=True, exist_ok=True)
  print(f"Downloading missing Lichess PGN: {url}")
  urllib.request.urlretrieve(url, path)
  print(f"Wrote: {path}")
  return path


def require_processed_files(paths, build_hint: str) -> None:
  """Raise a clear error when generated processed data files are absent."""
  missing = [Path(path) for path in paths if not Path(path).exists()]

  if not missing:
    return

  missing_lines = "\n".join(f"  - {path}" for path in missing)
  raise FileNotFoundError(
    "Missing processed data files:\n"
    f"{missing_lines}\n\n"
    "These files are generated artifacts and are not downloaded automatically.\n"
    f"Regenerate them with:\n{build_hint}"
  )
