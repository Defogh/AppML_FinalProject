from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import bz2
import gzip
import io
import re
from typing import TextIO

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regex objects are compiled once at import time.
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$')
BRACE_COMMENT_RE = re.compile(r"\{([^{}]*)\}")
SEMICOLON_COMMENT_RE = re.compile(r";[^\n\r]*")
MOVE_NUMBER_PREFIX_RE = re.compile(r"^\d+\.(?:\.\.)?")
CLOCK_RE = re.compile(r"\[%clkc?\s+([^\]\s]+)\s*\]")
EVAL_RE = re.compile(r"\[%eval\s+([^\]\s]+)\s*\]")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
LIST_SEP = "|"


# ---------------------------------------------------------------------------
# Small parsing helpers.
# ---------------------------------------------------------------------------

def parse_int(value: object) -> int | None:
    """Convert PGN integer-like values to ``int``; otherwise return ``None``."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def parse_result_score(result: str | None) -> float | None:
    """Return the game result from White's perspective."""
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result == "1/2-1/2":
        return 0.5
    return None


def parse_time_control(time_control: str | None) -> tuple[int | None, int | None]:
    """
    Parse Lichess-style time controls such as ``"180+0"``.

    Returns
    -------
    initial_seconds, increment_seconds
        ``None`` values are returned for unknown/non-clock formats such as ``"-"``.
    """
    if not time_control or time_control == "-":
        return None, None

    parts = time_control.split("+", maxsplit=1)
    if len(parts) != 2:
        return None, None

    initial = parse_int(parts[0])
    increment = parse_int(parts[1])
    return initial, increment


def classify_time_control(initial_seconds: int | None, increment_seconds: int | None) -> str | None:
    """
    Approximate Lichess speed category.

    Lichess commonly uses estimated game duration = initial + 40 * increment.
    """
    if initial_seconds is None or increment_seconds is None:
        return None

    estimated_seconds = initial_seconds + 40 * increment_seconds

    if estimated_seconds < 180:
        return "bullet"
    if estimated_seconds < 480:
        return "blitz"
    if estimated_seconds < 1500:
        return "rapid"
    return "classical"


def parse_clock_seconds(value: str | None) -> float | None:
    """
    Parse a Lichess ``%clk`` / ``%clkc`` value into seconds.

    Accepted examples are ``"0:00:30"``, ``"3:02"``, ``"0:00:30.42"`` and
    plain second values. ``None`` is returned for malformed values.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    parts = text.split(":")

    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return None

    return None


def parse_eval_token(value: str | None) -> tuple[float | None, int | None]:
    """
    Parse a Lichess ``%eval`` token.

    Returns
    -------
    eval_pawns, mate_in
        Centipawn-style evaluations are returned in pawn units, from White's
        perspective. Mate scores such as ``#-4`` are returned as ``mate_in`` and
        leave ``eval_pawns`` as ``None``.
    """
    if value is None:
        return None, None

    text = str(value).strip()
    if not text:
        return None, None

    if text.startswith("#"):
        return None, parse_int(text[1:])

    try:
        return float(text), None
    except ValueError:
        return None, None


def _format_scalar(value: object) -> str:
    """Format scalar values for compact pipe-separated storage."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def serialize_list(values: Sequence[object]) -> str:
    """Serialize a list-like object into a pipe-separated string for CSV output."""
    return LIST_SEP.join(_format_scalar(value) for value in values)


def extract_lichess_id(site: str | None) -> str | None:
    """Extract the game id from a Lichess game URL when possible."""
    if not site:
        return None
    return site.rstrip("/").split("/")[-1] or None


# ---------------------------------------------------------------------------
# Input handling.
# ---------------------------------------------------------------------------

def open_text_maybe_compressed(path: str | Path) -> TextIO:
    """
    Open plain text, .gz, .bz2, or .zst files as text.

    Lichess standard database downloads are normally ``.pgn.zst``. Support for
    that format requires ``pip install zstandard``.
    """
    path = Path(path)
    suffixes = path.suffixes

    if suffixes and suffixes[-1] == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")

    if suffixes and suffixes[-1] == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")

    if suffixes and suffixes[-1] == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "Reading .zst PGN files requires zstandard. Install it with: pip install zstandard"
            ) from exc

        compressed = path.open("rb")
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(compressed)
        # Attach the binary file to the TextIOWrapper so closing the wrapper
        # closes the decompressor stream. The underlying file is closed by the
        # stream reader when the wrapper is closed.
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")

    return path.open(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# PGN iteration and movetext parsing.
# ---------------------------------------------------------------------------

def iter_pgn_games(pgn_path: str | Path) -> Iterator[tuple[dict[str, str], str]]:
    """
    Yield ``(headers, movetext)`` pairs from a PGN file.

    This avoids constructing ``python-chess`` game trees. It is therefore much
    faster when the goal is metadata and SAN/comment-derived features.
    """
    headers: dict[str, str] = {}
    movetext_parts: list[str] = []

    with open_text_maybe_compressed(pgn_path) as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                if headers and movetext_parts:
                    yield headers, "\n".join(movetext_parts)
                    headers = {}
                    movetext_parts = []
                continue

            match = HEADER_RE.match(line)
            if match:
                key, value = match.groups()
                headers[key] = value
            else:
                if headers:
                    movetext_parts.append(line)

    if headers and movetext_parts:
        yield headers, "\n".join(movetext_parts)


def _remove_variations(text: str) -> str:
    """
    Remove parenthesized PGN variations.

    PGN variations can be nested, so repeatedly remove the innermost pairs.
    Lichess database exports normally contain only the mainline, but this keeps
    the parser from accidentally treating variation annotations as mainline data.
    """
    while "(" in text and ")" in text:
        new_text = re.sub(r"\([^()]*\)", " ", text)
        if new_text == text:
            break
        text = new_text
    return text


def _normalize_san_token(raw_token: str) -> str | None:
    """Normalize one raw PGN token into a SAN token, or return ``None``."""
    token = raw_token.strip()
    if not token:
        return None

    # Handle both ``1. e4`` and compact forms such as ``1.e4`` / ``1...c5``.
    token = MOVE_NUMBER_PREFIX_RE.sub("", token)
    if not token or token == "...":
        return None

    if token in RESULT_TOKENS or token.startswith("$"):
        return None

    # Keep SAN symbols such as x, +, # and =Q, but remove human annotations.
    token = token.rstrip("!?")
    return token or None


def parse_movetext(movetext: str) -> tuple[list[str], list[float | None], list[float | None], list[int | None]]:
    """
    Parse PGN movetext into SAN tokens and per-ply Lichess annotations.

    Returns
    -------
    san, clock_seconds, eval_pawns, eval_mate
        All four lists have the same length. Missing annotations are stored as
        ``None`` at the corresponding ply.
    """
    text = SEMICOLON_COMMENT_RE.sub(" ", movetext)
    text = _remove_variations(text)

    san: list[str] = []
    clock_seconds: list[float | None] = []
    eval_pawns: list[float | None] = []
    eval_mate: list[int | None] = []

    def append_tokens(segment: str) -> None:
        for raw_token in segment.split():
            token = _normalize_san_token(raw_token)
            if token is None:
                continue
            san.append(token)
            clock_seconds.append(None)
            eval_pawns.append(None)
            eval_mate.append(None)

    position = 0
    for comment_match in BRACE_COMMENT_RE.finditer(text):
        append_tokens(text[position:comment_match.start()])

        # Lichess comments are attached to the preceding move. Multiple [%...]
        # tags can appear in the same comment, e.g. [%eval ...] [%clk ...].
        if san:
            comment = comment_match.group(1)

            clock_match = CLOCK_RE.search(comment)
            if clock_match:
                clock_seconds[-1] = parse_clock_seconds(clock_match.group(1))

            eval_match = EVAL_RE.search(comment)
            if eval_match:
                eval_cp, mate_in = parse_eval_token(eval_match.group(1))
                eval_pawns[-1] = eval_cp
                eval_mate[-1] = mate_in

        position = comment_match.end()

    append_tokens(text[position:])

    return san, clock_seconds, eval_pawns, eval_mate


def san_tokens(movetext: str) -> list[str]:
    """Return only the normalized SAN token list from PGN movetext."""
    san, _, _, _ = parse_movetext(movetext)
    return san


# ---------------------------------------------------------------------------
# Game-level conversion.
# ---------------------------------------------------------------------------

def game_to_row(
    headers: dict[str, str],
    movetext: str,
    *,
    game_index: int | None = None,
    include_moves_san: bool = True,
    include_annotation_series: bool = True,
    include_raw_movetext: bool = False,
) -> dict[str, object] | None:
    """
    Convert one PGN game into one raw CSV/dataframe row.

    This is the only ingestion target for the workflow: one row per game. It
    preserves the fields needed later for both player summary features and SAN
    autoencoder features.
    """
    white = headers.get("White")
    black = headers.get("Black")
    white_elo = parse_int(headers.get("WhiteElo"))
    black_elo = parse_int(headers.get("BlackElo"))

    if white in (None, "?") or black in (None, "?"):
        return None

    if white_elo is None or black_elo is None:
        return None

    san, clock_seconds, eval_pawns, eval_mate = parse_movetext(movetext)
    initial_seconds, increment_seconds = parse_time_control(headers.get("TimeControl"))
    elo_diff = white_elo - black_elo

    num_halfmoves = len(san)
    num_clock_annotations = sum(value is not None for value in clock_seconds)
    num_eval_annotations = sum((cp is not None) or (mate is not None) for cp, mate in zip(eval_pawns, eval_mate))

    row: dict[str, object] = {
        "game_id": game_index,
        "lichess_id": extract_lichess_id(headers.get("Site")),
        "event": headers.get("Event"),
        "site": headers.get("Site"),
        "date": headers.get("UTCDate") or headers.get("Date"),
        "time": headers.get("UTCTime"),
        "round": headers.get("Round"),
        "white": white,
        "black": black,
        "result": headers.get("Result"),
        "result_white_score": parse_result_score(headers.get("Result")),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "avg_elo": (white_elo + black_elo) / 2,
        "elo_diff": elo_diff,
        "abs_elo_diff": abs(elo_diff),
        "white_rating_diff": parse_int(headers.get("WhiteRatingDiff")),
        "black_rating_diff": parse_int(headers.get("BlackRatingDiff")),
        "white_title": headers.get("WhiteTitle"),
        "black_title": headers.get("BlackTitle"),
        "eco": headers.get("ECO"),
        "eco_family": (headers.get("ECO") or "")[:1] or None,
        "opening": headers.get("Opening"),
        "time_control": headers.get("TimeControl"),
        "initial_seconds": initial_seconds,
        "increment_seconds": increment_seconds,
        "speed": classify_time_control(initial_seconds, increment_seconds),
        "termination": headers.get("Termination"),
        "variant": headers.get("Variant"),
        "num_halfmoves": num_halfmoves,
        "num_fullmoves": (num_halfmoves + 1) // 2,
        "num_white_moves": (num_halfmoves + 1) // 2,
        "num_black_moves": num_halfmoves // 2,
        "num_clock_annotations": num_clock_annotations,
        "clock_coverage": num_clock_annotations / num_halfmoves if num_halfmoves else 0.0,
        "num_eval_annotations": num_eval_annotations,
        "eval_coverage": num_eval_annotations / num_halfmoves if num_halfmoves else 0.0,
    }

    if include_moves_san:
        row["moves_san"] = " ".join(san)

    if include_annotation_series:
        row["clock_seconds_by_ply"] = serialize_list(clock_seconds)
        row["eval_pawns_by_ply"] = serialize_list(eval_pawns)
        row["eval_mate_by_ply"] = serialize_list(eval_mate)

    if include_raw_movetext:
        row["movetext"] = movetext

    return row


def iter_rows(
    pgn_path: str | Path,
    *,
    include_moves_san: bool = True,
    include_annotation_series: bool = True,
    include_raw_movetext: bool = False,
    max_games: int | None = None,
) -> Iterator[dict[str, object]]:
    """Stream parsed one-row-per-game dictionaries from a PGN file."""
    n_rows = 0
    for game_idx, (headers, movetext) in enumerate(iter_pgn_games(pgn_path)):
        if max_games is not None and n_rows >= max_games:
            break

        row = game_to_row(
            headers,
            movetext,
            game_index=game_idx,
            include_moves_san=include_moves_san,
            include_annotation_series=include_annotation_series,
            include_raw_movetext=include_raw_movetext,
        )
        if row is None:
            continue

        n_rows += 1
        yield row


def parser(
    pgn_path: str | Path,
    *,
    include_moves_san: bool = True,
    include_annotation_series: bool = True,
    include_raw_movetext: bool = False,
    max_games: int | None = None,
) -> pd.DataFrame:
    """
    Parse a PGN file into one raw row per game.

    For large full-month Lichess files, prefer ``convert_pgn_to_csv`` because it
    writes chunks to disk instead of holding all rows in memory.
    """
    return pd.DataFrame(iter_rows(
        pgn_path,
        include_moves_san=include_moves_san,
        include_annotation_series=include_annotation_series,
        include_raw_movetext=include_raw_movetext,
        max_games=max_games,
    ))


def convert_pgn_to_csv(
    pgn_path: str | Path,
    csv_path: str | Path,
    *,
    sample_csv_path: str | Path | None = None,
    sample_size: int = 300_000,
    random_state: int = 42,
    chunk_size: int = 100_000,
    include_moves_san: bool = True,
    include_annotation_series: bool = True,
    include_raw_movetext: bool = False,
    max_games: int | None = None,
) -> dict[str, int | str | None]:
    """
    Convert PGN to a one-row-per-game CSV and optionally write a random sample.

    The sample is drawn by reservoir sampling during the PGN scan, so it does
    not require reading the completed full CSV back into memory.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    sample_path = Path(sample_csv_path) if sample_csv_path is not None else None
    if sample_path is not None:
        sample_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)
    reservoir: list[dict[str, object]] = []
    buffer: list[dict[str, object]] = []
    n_rows = 0
    wrote_header = False

    for row in iter_rows(
        pgn_path,
        include_moves_san=include_moves_san,
        include_annotation_series=include_annotation_series,
        include_raw_movetext=include_raw_movetext,
        max_games=max_games,
    ):
        buffer.append(row)
        n_rows += 1

        if sample_path is not None and sample_size > 0:
            if len(reservoir) < sample_size:
                reservoir.append(row.copy())
            else:
                replace_idx = int(rng.integers(0, n_rows))
                if replace_idx < sample_size:
                    reservoir[replace_idx] = row.copy()

        if len(buffer) >= chunk_size:
            pd.DataFrame(buffer).to_csv(
                csv_path,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            buffer.clear()

    if buffer:
        pd.DataFrame(buffer).to_csv(
            csv_path,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True

    if sample_path is not None:
        sample_df = pd.DataFrame(reservoir)
        if len(sample_df):
            sample_df = sample_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        sample_df.to_csv(sample_path, index=False)

    return {
        "rows_written": n_rows,
        "full_csv_path": str(csv_path),
        "sample_csv_path": str(sample_path) if sample_path is not None else None,
        "sample_rows_written": min(n_rows, sample_size) if sample_path is not None else 0,
    }


def sample_csv(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    sample_size: int = 300_000,
    random_state: int = 42,
    chunksize: int = 100_000,
) -> dict[str, int | str]:
    """
    Uniformly sample rows from an already-converted CSV using reservoir sampling.

    This is useful when the full conversion has already been done and you only
    want to regenerate the training subset.
    """
    rng = np.random.default_rng(random_state)
    reservoir: list[pd.Series] = []
    n_rows = 0

    for chunk in pd.read_csv(input_csv_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            n_rows += 1
            if len(reservoir) < sample_size:
                reservoir.append(row.copy())
            else:
                replace_idx = int(rng.integers(0, n_rows))
                if replace_idx < sample_size:
                    reservoir[replace_idx] = row.copy()

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df = pd.DataFrame(reservoir)
    if len(sample_df):
        sample_df = sample_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sample_df.to_csv(output_csv_path, index=False)

    return {
        "rows_seen": n_rows,
        "sample_rows_written": len(sample_df),
        "sample_csv_path": str(output_csv_path),
    }



# ---------------------------------------------------------------------------
# Workflow-specific converter.
# ---------------------------------------------------------------------------

def _rename_movetext_to_moves_pgn(row: dict[str, object]) -> dict[str, object]:
    """Use the project-facing column name ``moves_pgn`` for raw PGN movetext."""
    if "movetext" in row:
        row["moves_pgn"] = row.pop("movetext")
    return row


def convert_pgn_to_full_and_sample_csv(
    pgn_path: str | Path,
    full_csv_path: str | Path,
    sample_csv_path: str | Path,
    *,
    sample_size: int = 300_000,
    random_state: int = 42,
    chunk_size: int = 100_000,
    full_include_moves_san: bool = True,
    full_include_annotation_series: bool = True,
    full_include_moves_pgn: bool = False,
    sample_include_moves_san: bool = True,
    sample_include_annotation_series: bool = True,
    sample_include_moves_pgn: bool = True,
    max_games: int | None = None,
) -> dict[str, int | str]:
    """
    Convert a PGN file to a full raw CSV and a random training-sample CSV.

    The full CSV can be kept relatively compact, while the sample CSV can keep
    ``moves_pgn`` for slower board-aware feature extraction with python-chess.
    Sampling is uniform reservoir sampling over successfully parsed games and is
    performed during the same scan as the full conversion.

    Typical project setup
    ---------------------
    - full CSV: metadata + normalized ``moves_san`` + compact annotation series
    - sample CSV: same columns plus ``moves_pgn`` so board-aware features can be
      computed on the 300K training subset without bloating the full CSV
    """
    full_csv_path = Path(full_csv_path)
    sample_csv_path = Path(sample_csv_path)
    full_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)
    reservoir: list[dict[str, object]] = []
    full_buffer: list[dict[str, object]] = []
    n_rows = 0
    wrote_header = False

    for game_idx, (headers, movetext) in enumerate(iter_pgn_games(pgn_path)):
        if max_games is not None and n_rows >= max_games:
            break

        full_row = game_to_row(
            headers,
            movetext,
            game_index=game_idx,
            include_moves_san=full_include_moves_san,
            include_annotation_series=full_include_annotation_series,
            include_raw_movetext=full_include_moves_pgn,
        )
        if full_row is None:
            continue
        full_row = _rename_movetext_to_moves_pgn(full_row)

        sample_row = game_to_row(
            headers,
            movetext,
            game_index=game_idx,
            include_moves_san=sample_include_moves_san,
            include_annotation_series=sample_include_annotation_series,
            include_raw_movetext=sample_include_moves_pgn,
        )
        if sample_row is None:
            continue
        sample_row = _rename_movetext_to_moves_pgn(sample_row)

        full_buffer.append(full_row)
        n_rows += 1

        if len(reservoir) < sample_size:
            reservoir.append(sample_row)
        else:
            j = int(rng.integers(0, n_rows))
            if j < sample_size:
                reservoir[j] = sample_row

        if len(full_buffer) >= chunk_size:
            pd.DataFrame(full_buffer).to_csv(
                full_csv_path,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            full_buffer.clear()

    if full_buffer or not wrote_header:
        pd.DataFrame(full_buffer).to_csv(
            full_csv_path,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
        )

    pd.DataFrame(reservoir).to_csv(sample_csv_path, index=False)

    return {
        "input_path": str(pgn_path),
        "full_csv_path": str(full_csv_path),
        "sample_csv_path": str(sample_csv_path),
        "parsed_games": n_rows,
        "sample_games": len(reservoir),
        "random_state": random_state,
    }
