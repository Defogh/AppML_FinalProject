from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# Regex objects are compiled once at import time.
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$')
BRACE_COMMENT_RE = re.compile(r"\{[^{}]*\}")
SEMICOLON_COMMENT_RE = re.compile(r";[^\n\r]*")
MOVE_NUMBER_PREFIX_RE = re.compile(r"^\d+\.(?:\.\.)?")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

PIECE_LETTERS = frozenset("KQRBN")
PROMOTION_RE = re.compile(r"=([QRBN])")


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


def iter_pgn_games(pgn_path: str | Path) -> Iterator[tuple[dict[str, str], str]]:
    """
    Yield ``(headers, movetext)`` pairs from a PGN file.

    This avoids constructing ``python-chess`` game trees. It is therefore much
    faster when the goal is metadata and SAN-token-derived features.
    """
    pgn_path = Path(pgn_path)

    headers: dict[str, str] = {}
    movetext_parts: list[str] = []

    with pgn_path.open(encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                if headers and movetext_parts:
                    yield headers, " ".join(movetext_parts)
                    headers = {}
                    movetext_parts = []
                continue

            match = HEADER_RE.match(line)
            if match:
                key, value = match.groups()
                headers[key] = value
            else:
                # This line belongs to the game movetext. Standard PGN has a
                # blank line between the headers and the movetext, but Lichess
                # files may still contain long movetext spanning multiple lines.
                if headers:
                    movetext_parts.append(line)

    if headers and movetext_parts:
        yield headers, " ".join(movetext_parts)


def _remove_variations(text: str) -> str:
    """
    Remove parenthesized PGN variations.

    PGN variations can be nested, so repeatedly remove the innermost pairs.
    For Lichess database exports this is usually a no-op because games normally
    contain only the mainline.
    """
    while "(" in text and ")" in text:
        new_text = re.sub(r"\([^()]*\)", " ", text)
        if new_text == text:
            break
        text = new_text
    return text


def san_tokens(movetext: str) -> list[str]:
    """
    Convert PGN movetext to a normalized SAN-token list.

    The output keeps chess symbols such as ``x``, ``+``, ``#`` and ``=Q``,
    but removes move numbers, game results, NAGs, comments and common
    annotation suffixes such as ``!`` and ``?``.
    """
    text = BRACE_COMMENT_RE.sub(" ", movetext)
    text = SEMICOLON_COMMENT_RE.sub(" ", text)
    text = _remove_variations(text)

    tokens: list[str] = []

    for raw_token in text.split():
        token = raw_token.strip()
        if not token:
            continue

        # Handle both "1. e4" and compact "1.e4" forms.
        token = MOVE_NUMBER_PREFIX_RE.sub("", token)
        if not token:
            continue

        if token in RESULT_TOKENS or token.startswith("$"):
            continue

        token = token.rstrip("!?")
        if token:
            tokens.append(token)

    return tokens


def extract_move_features(tokens: list[str]) -> dict[str, object]:
    """
    Extract syntactic chess features from SAN tokens.

    These features do not require legal move reconstruction. They are therefore
    fast, but they should be understood as SAN-string features rather than
    board-state-derived features.
    """
    num_halfmoves = len(tokens)

    num_captures = 0
    white_captures = 0
    black_captures = 0

    num_checks = 0
    white_checks = 0
    black_checks = 0

    num_checkmates = 0
    white_checkmates = 0
    black_checkmates = 0

    num_promotions = 0
    num_castles_kingside = 0
    num_castles_queenside = 0

    white_castled = False
    black_castled = False
    white_castle_side: str | None = None
    black_castle_side: str | None = None

    num_pawn_moves = 0
    num_piece_moves = 0
    num_king_moves = 0
    num_queen_moves = 0
    num_rook_moves = 0
    num_bishop_moves = 0
    num_knight_moves = 0

    num_queen_promotions = 0
    num_rook_promotions = 0
    num_bishop_promotions = 0
    num_knight_promotions = 0

    for ply, token in enumerate(tokens):
        is_white = (ply % 2) == 0

        # Castling first, because "O-O" is not a normal piece move.
        is_queenside_castle = token.startswith("O-O-O") or token.startswith("0-0-0")
        is_kingside_castle = (
            not is_queenside_castle
            and (token.startswith("O-O") or token.startswith("0-0"))
        )

        if is_queenside_castle or is_kingside_castle:
            if is_queenside_castle:
                num_castles_queenside += 1
                side = "queenside"
            else:
                num_castles_kingside += 1
                side = "kingside"

            if is_white:
                white_castled = True
                white_castle_side = side
            else:
                black_castled = True
                black_castle_side = side
        else:
            first = token[0]
            if first in PIECE_LETTERS:
                num_piece_moves += 1
                if first == "K":
                    num_king_moves += 1
                elif first == "Q":
                    num_queen_moves += 1
                elif first == "R":
                    num_rook_moves += 1
                elif first == "B":
                    num_bishop_moves += 1
                elif first == "N":
                    num_knight_moves += 1
            else:
                num_pawn_moves += 1

        if "x" in token:
            num_captures += 1
            if is_white:
                white_captures += 1
            else:
                black_captures += 1

        if "+" in token:
            num_checks += 1
            if is_white:
                white_checks += 1
            else:
                black_checks += 1

        if "#" in token:
            num_checkmates += 1
            if is_white:
                white_checkmates += 1
            else:
                black_checkmates += 1

        promotion_match = PROMOTION_RE.search(token)
        if promotion_match:
            num_promotions += 1
            promotion_piece = promotion_match.group(1)
            if promotion_piece == "Q":
                num_queen_promotions += 1
            elif promotion_piece == "R":
                num_rook_promotions += 1
            elif promotion_piece == "B":
                num_bishop_promotions += 1
            elif promotion_piece == "N":
                num_knight_promotions += 1

    return {
        "num_halfmoves": num_halfmoves,
        "num_moves": num_halfmoves / 2,
        "num_fullmoves": (num_halfmoves + 1) // 2,
        "num_white_moves": (num_halfmoves + 1) // 2,
        "num_black_moves": num_halfmoves // 2,
        "num_captures": num_captures,
        "white_captures": white_captures,
        "black_captures": black_captures,
        "num_checks": num_checks,
        "white_checks": white_checks,
        "black_checks": black_checks,
        "num_checkmates": num_checkmates,
        "white_checkmates": white_checkmates,
        "black_checkmates": black_checkmates,
        "num_promotions": num_promotions,
        "num_queen_promotions": num_queen_promotions,
        "num_rook_promotions": num_rook_promotions,
        "num_bishop_promotions": num_bishop_promotions,
        "num_knight_promotions": num_knight_promotions,
        "num_castles_kingside": num_castles_kingside,
        "num_castles_queenside": num_castles_queenside,
        "white_castled": white_castled,
        "black_castled": black_castled,
        "white_castle_side": white_castle_side,
        "black_castle_side": black_castle_side,
        "num_pawn_moves": num_pawn_moves,
        "num_piece_moves": num_piece_moves,
        "num_king_moves": num_king_moves,
        "num_queen_moves": num_queen_moves,
        "num_rook_moves": num_rook_moves,
        "num_bishop_moves": num_bishop_moves,
        "num_knight_moves": num_knight_moves,
        "first_white_move": tokens[0] if num_halfmoves >= 1 else None,
        "first_black_move": tokens[1] if num_halfmoves >= 2 else None,
        "last_move": tokens[-1] if num_halfmoves else None,
        "opening_san_4ply": " ".join(tokens[:4]),
        "opening_san_6ply": " ".join(tokens[:6]),
        "opening_san_10ply": " ".join(tokens[:10]),
        "moves_san": " ".join(tokens),
    }


def parser(
    pgn_path: str | Path,
    *,
    include_moves_san: bool = True,
    max_games: int | None = None,
) -> pd.DataFrame:
    """
    Parse a PGN file into a dataframe.

    This parser is optimized for large Lichess PGN exports. It preserves the
    original metadata columns from the old parser and adds derived quantities
    from the SAN movetext.

    Parameters
    ----------
    pgn_path:
        Path to the PGN file.
    include_moves_san:
        If ``False``, omit the full normalized move string to save memory.
        Opening-prefix columns are still kept.
    max_games:
        Optional cap, useful for quick tests/benchmarks.
    """
    games: list[dict[str, object]] = []

    for game_idx, (headers, movetext) in enumerate(iter_pgn_games(pgn_path)):
        if max_games is not None and game_idx >= max_games:
            break

        white = headers.get("White")
        black = headers.get("Black")
        white_elo = parse_int(headers.get("WhiteElo"))
        black_elo = parse_int(headers.get("BlackElo"))

        if white in (None, "?") or black in (None, "?"):
            continue

        if white_elo is None or black_elo is None:
            continue

        tokens = san_tokens(movetext)
        move_features = extract_move_features(tokens)

        if not include_moves_san:
            move_features.pop("moves_san", None)

        initial_seconds, increment_seconds = parse_time_control(headers.get("TimeControl"))
        elo_diff = white_elo - black_elo

        games.append(
            {
                "event": headers.get("Event"),
                "site": headers.get("Site"),
                "white": white,
                "black": black,
                "result": headers.get("Result"),
                "result_white_score": parse_result_score(headers.get("Result")),
                "date": headers.get("UTCDate"),
                "time": headers.get("UTCTime"),
                "white_elo": white_elo,
                "black_elo": black_elo,
                "avg_elo": (white_elo + black_elo) / 2,
                "elo_diff": elo_diff,
                "abs_elo_diff": abs(elo_diff),
                "white_rating_diff": parse_int(headers.get("WhiteRatingDiff")),
                "black_rating_diff": parse_int(headers.get("BlackRatingDiff")),
                "eco": headers.get("ECO"),
                "eco_family": (headers.get("ECO") or "")[:1] or None,
                "opening": headers.get("Opening"),
                "time_control": headers.get("TimeControl"),
                "initial_seconds": initial_seconds,
                "increment_seconds": increment_seconds,
                "speed": classify_time_control(initial_seconds, increment_seconds),
                "termination": headers.get("Termination"),
                **move_features,
            }
        )

    return pd.DataFrame(games)