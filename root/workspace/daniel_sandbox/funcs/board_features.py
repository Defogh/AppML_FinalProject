"""
Board-aware chess feature extraction for the project workflow.

This module adapts the useful part of chess_features_final.py: it replays each
mainline on a real python-chess board and returns board-state-derived features.
It is intentionally separate from PGN -> CSV ingestion, because board replay is
much slower than the regex streaming parser and should usually be run on the
300K training sample, not necessarily the full raw dataset.

Expected input columns, in priority order:
    moves_pgn   raw PGN movetext with comments, best for eval-derived features
    movetext    alias for raw PGN movetext
    moves_san   normalized SAN string, good for board features but no comments
    Moves       legacy column name

Install dependencies:
    pip install chess joblib
"""

from __future__ import annotations

import io
import re
from typing import Optional

import numpy as np
import pandas as pd

_EVAL_RE = re.compile(r"\[%eval\s+([^\]]+)\]")
_RESULT_AT_END_RE = re.compile(r"\s+(1-0|0-1|1/2-1/2|\*)\s*$")


def _require_chess():
    try:
        import chess
        import chess.pgn
    except ImportError as exc:
        raise ImportError(
            "Board-aware features require python-chess. Install it with: pip install chess"
        ) from exc
    return chess


def _eval_to_cp(value: str) -> Optional[int]:
    """Convert a Lichess [%eval ...] value to centipawns; mate scores are clamped."""
    if not value:
        return None

    text = str(value).strip()
    if text.startswith("#"):
        try:
            mate = int(text[1:])
        except ValueError:
            return None
        return 10_000 if mate > 0 else -10_000

    try:
        return int(float(text) * 100)
    except ValueError:
        return None


def _parse_game(moves_string: object, result_string: str = "*"):
    """
    Parse raw PGN movetext or a bare SAN string into a python-chess Game.

    The parser is deliberately tolerant: it accepts Lichess comments such as
    [%clk ...] / [%eval ...], move numbers, or a plain normalized SAN sequence.
    """
    chess = _require_chess()

    if moves_string is None or (isinstance(moves_string, float) and np.isnan(moves_string)):
        return None

    text = str(moves_string).strip()
    if not text:
        return None

    text = _RESULT_AT_END_RE.sub("", text)
    result = result_string if result_string in {"1-0", "0-1", "1/2-1/2", "*"} else "*"

    pgn_text = f'[Event "?"]\n[Result "{result}"]\n\n{text} {result}\n'
    try:
        return chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return None


def _empty_board_features() -> dict[str, object]:
    numeric = [
        "total_ply_count", "material_balance_end", "result_encoded",
        "checks_given_white", "checks_given_black",
        "check_density_white", "check_density_black",
        "first_capture_move_white", "first_capture_move_black",
        "pawn_captures_total", "piece_captures_total", "capture_density",
        "castle_move_white", "castle_move_black",
        "consec_same_piece_white", "consec_same_piece_black",
        "queen_moves_before_10", "white_territory_depth", "black_territory_depth",
        "promotions", "en_passant_captures",
        "legal_moves_white_move5", "legal_moves_black_move5",
        "acpl_white", "inaccuracy_count_white", "mistake_count_white",
        "blunder_count_white", "blunder_density_white",
        "acpl_black", "inaccuracy_count_black", "mistake_count_black",
        "blunder_count_black", "blunder_density_black",
    ]
    out = {column: float("nan") for column in numeric}
    out["castle_side_white"] = "none"
    out["castle_side_black"] = "none"
    return out


def extract_board_features(moves_string: object, result_string: str = "*") -> dict[str, object]:
    """
    Replay one game and compute board-aware features.

    This is based on the friend implementation, with project-specific changes:
    tolerant input column handling is done outside this function, dependency
    imports are lazy, and the parser accepts both raw Lichess movetext and the
    normalized ``moves_san`` column emitted by the fast PGN converter.
    """
    chess = _require_chess()
    game = _parse_game(moves_string, result_string)
    if game is None:
        return _empty_board_features()

    board = game.board()
    moves = list(game.mainline_moves())

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    checks_w = checks_b = 0
    first_cap_w = first_cap_b = 0
    pawn_caps = piece_caps = 0
    castle_move_w = castle_move_b = 0
    castle_side_w = castle_side_b = "none"
    consec_w = consec_b = 0
    queen_before_10 = 0
    max_rank_w = 0
    min_rank_b = 7
    promotions = en_passant = 0
    legal5_w = legal5_b = 0
    last_dest_w = last_dest_b = None

    cp_losses_w: list[int] = []
    cp_losses_b: list[int] = []
    prev_eval: Optional[int] = 20

    node = game
    while node.variations:
        node = node.variations[0]
        move = node.move
        comment = node.comment
        fullmove_number = board.fullmove_number
        is_white = board.turn == chess.WHITE

        if fullmove_number == 5:
            if is_white and legal5_w == 0:
                legal5_w = board.legal_moves.count()
            elif not is_white and legal5_b == 0:
                legal5_b = board.legal_moves.count()

        m_eval = _EVAL_RE.search(comment)
        if m_eval:
            current_eval = _eval_to_cp(m_eval.group(1))
            if current_eval is not None and prev_eval is not None:
                # Evaluation is from White's perspective. A loss is the change
                # against the side that just moved.
                loss = (prev_eval - current_eval) if is_white else (current_eval - prev_eval)
                loss = max(0, min(1000, loss))
                (cp_losses_w if is_white else cp_losses_b).append(loss)
            prev_eval = current_eval
        else:
            prev_eval = None

        if board.is_en_passant(move):
            en_passant += 1

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece is None or captured_piece.piece_type == chess.PAWN:
                pawn_caps += 1
            else:
                piece_caps += 1
            if is_white and first_cap_w == 0:
                first_cap_w = fullmove_number
            if (not is_white) and first_cap_b == 0:
                first_cap_b = fullmove_number

        if board.is_castling(move):
            side = "king" if board.is_kingside_castling(move) else "queen"
            if is_white:
                castle_move_w = fullmove_number
                castle_side_w = side
            else:
                castle_move_b = fullmove_number
                castle_side_b = side

        if move.promotion is not None:
            promotions += 1

        destination_rank = chess.square_rank(move.to_square)
        if is_white:
            max_rank_w = max(max_rank_w, destination_rank)
        else:
            min_rank_b = min(min_rank_b, destination_rank)

        if fullmove_number < 10:
            moved_piece = board.piece_at(move.from_square)
            if moved_piece and moved_piece.piece_type == chess.QUEEN:
                queen_before_10 += 1

        if is_white:
            if last_dest_w is not None and move.from_square == last_dest_w:
                consec_w += 1
            last_dest_w = move.to_square
        else:
            if last_dest_b is not None and move.from_square == last_dest_b:
                consec_b += 1
            last_dest_b = move.to_square

        board.push(move)

        if board.is_check():
            if is_white:
                checks_w += 1
            else:
                checks_b += 1

    total_ply = len(moves)
    white_material = sum(piece_values[pt] * len(board.pieces(pt, chess.WHITE)) for pt in piece_values)
    black_material = sum(piece_values[pt] * len(board.pieces(pt, chess.BLACK)) for pt in piece_values)

    result_encoded = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "*": 0}.get(result_string, 0)

    def _quality_stats(losses: list[int]) -> tuple[float, float, float, float]:
        if not losses:
            return float("nan"), float("nan"), float("nan"), float("nan")
        arr = np.array(losses)
        return (
            float(arr.mean()),
            float(np.sum((arr >= 50) & (arr < 100))),
            float(np.sum((arr >= 100) & (arr < 300))),
            float(np.sum(arr >= 300)),
        )

    acpl_w, inacc_w, mist_w, blun_w = _quality_stats(cp_losses_w)
    acpl_b, inacc_b, mist_b, blun_b = _quality_stats(cp_losses_b)

    total_den = max(1, total_ply)
    white_den = max(1, total_ply // 2 + total_ply % 2)
    black_den = max(1, total_ply // 2)

    return {
        "total_ply_count": total_ply,
        "material_balance_end": white_material - black_material,
        "result_encoded": result_encoded,
        "checks_given_white": checks_w,
        "checks_given_black": checks_b,
        "check_density_white": checks_w / white_den,
        "check_density_black": checks_b / black_den,
        "first_capture_move_white": first_cap_w,
        "first_capture_move_black": first_cap_b,
        "pawn_captures_total": pawn_caps,
        "piece_captures_total": piece_caps,
        "capture_density": (pawn_caps + piece_caps) / total_den,
        "castle_move_white": castle_move_w,
        "castle_move_black": castle_move_b,
        "castle_side_white": castle_side_w,
        "castle_side_black": castle_side_b,
        "consec_same_piece_white": consec_w,
        "consec_same_piece_black": consec_b,
        "queen_moves_before_10": queen_before_10,
        "white_territory_depth": max(0, max_rank_w - 3),
        "black_territory_depth": max(0, 4 - min_rank_b),
        "promotions": promotions,
        "en_passant_captures": en_passant,
        "legal_moves_white_move5": legal5_w,
        "legal_moves_black_move5": legal5_b,
        "acpl_white": acpl_w,
        "inaccuracy_count_white": inacc_w,
        "mistake_count_white": mist_w,
        "blunder_count_white": blun_w,
        "blunder_density_white": blun_w / white_den if not np.isnan(blun_w) else float("nan"),
        "acpl_black": acpl_b,
        "inaccuracy_count_black": inacc_b,
        "mistake_count_black": mist_b,
        "blunder_count_black": blun_b,
        "blunder_density_black": blun_b / black_den if not np.isnan(blun_b) else float("nan"),
    }


def _resolve_moves_column(df: pd.DataFrame, moves_col: str | None) -> str:
    if moves_col is not None:
        if moves_col not in df:
            raise ValueError(f"Requested moves_col={moves_col!r}, but it is not in the dataframe.")
        return moves_col

    for candidate in ["moves_pgn", "movetext", "moves_san", "Moves"]:
        if candidate in df:
            return candidate

    raise ValueError("No move column found. Expected one of: moves_pgn, movetext, moves_san, Moves.")


def _resolve_result_column(df: pd.DataFrame, result_col: str | None) -> str | None:
    if result_col is not None:
        if result_col not in df:
            raise ValueError(f"Requested result_col={result_col!r}, but it is not in the dataframe.")
        return result_col

    for candidate in ["result", "Result"]:
        if candidate in df:
            return candidate

    return None


def _extract_row(moves: object, result: str) -> dict[str, object]:
    return extract_board_features(moves, result)


def extract_board_features_dataframe(
    df: pd.DataFrame,
    *,
    moves_col: str | None = None,
    result_col: str | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Extract board-aware features for every row in a dataframe.

    For large samples, use ``n_jobs=-1``. For debugging, use ``n_jobs=1`` so
    tracebacks are easier to read.
    """
    moves_col = _resolve_moves_column(df, moves_col)
    result_col = _resolve_result_column(df, result_col)

    moves = df[moves_col].tolist()
    results = df[result_col].fillna("*").astype(str).tolist() if result_col else ["*"] * len(df)

    if n_jobs == 1:
        records = [_extract_row(m, r) for m, r in zip(moves, results)]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError("Parallel board extraction requires joblib. Install it with: pip install joblib") from exc

        records = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_extract_row)(m, r) for m, r in zip(moves, results)
        )

    return pd.DataFrame.from_records(records, index=df.index)


def compute_elo_sample_weights(
    elo_series: pd.Series,
    bins: list[int] | None = None,
) -> np.ndarray:
    """
    Inverse-frequency weights by Elo stratum.

    Useful if you later train supervised Elo models or autoencoder variants where
    rare high-Elo examples should not be swamped by common rating ranges.
    """
    if bins is None:
        bins = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 4000]

    labels = pd.cut(elo_series, bins=bins, labels=False, right=True)
    counts = labels.value_counts()
    inv_freq = 1.0 / counts
    weights = labels.map(inv_freq).astype(float).fillna(1.0).to_numpy()
    return weights / np.nanmean(weights)
