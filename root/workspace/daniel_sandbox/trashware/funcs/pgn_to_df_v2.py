from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import re

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
# PGN iteration and movetext parsing.
# ---------------------------------------------------------------------------

def iter_pgn_games(pgn_path: str | Path) -> Iterator[tuple[dict[str, str], str]]:
    """
    Yield ``(headers, movetext)`` pairs from a PGN file.

    This avoids constructing ``python-chess`` game trees. It is therefore much
    faster when the goal is metadata and SAN/comment-derived features.
    """
    pgn_path = Path(pgn_path)

    headers: dict[str, str] = {}
    movetext_parts: list[str] = []

    with pgn_path.open(encoding="utf-8", errors="replace") as file:
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
    Convert one PGN game into one raw dataframe row.

    The row intentionally keeps mostly metadata and serialized mainline data.
    Heavier ML feature construction is left to ``parse_to_game_data`` and
    ``parse_to_player_data`` in ``clustering.py``.
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

    This parser is optimized for large Lichess PGN exports. It keeps metadata,
    normalized SAN, and serialized per-ply ``%clk`` / ``%eval`` annotations.
    Downstream ML-oriented feature tables should be built with the functions in
    ``clustering.py``.
    """
    rows: list[dict[str, object]] = []

    for game_idx, (headers, movetext) in enumerate(iter_pgn_games(pgn_path)):
        if max_games is not None and len(rows) >= max_games:
            break

        row = game_to_row(
            headers,
            movetext,
            game_index=game_idx,
            include_moves_san=include_moves_san,
            include_annotation_series=include_annotation_series,
            include_raw_movetext=include_raw_movetext,
        )
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)
