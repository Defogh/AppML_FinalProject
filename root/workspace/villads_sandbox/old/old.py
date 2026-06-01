"""
chess_features.py
-----------------
Converts a raw PGN moves string (as stored in the Lichess dataset) into a
rich feature vector by replaying the game on a real chess board via the
`python-chess` library.

Key design decisions
--------------------
* We parse the PGN moves string with `chess.pgn.read_game`, which gives us
  a proper Game tree with full board state at every ply.
* Legal-move counts are obtained from `board.legal_moves` *before* the
  move is pushed — the only way to get the correct count without heuristics.
* Castling detection checks the special move type flags that python-chess
  exposes rather than trying to pattern-match SAN strings.
* Material balance uses standard piece values (Q=9, R=5, B=3, N=3, P=1).
* "Enemy territory" for White is ranks 5-8 (indices 4-7), for Black it's
  ranks 1-4 (indices 0-3).
* En passant is detected via `board.is_en_passant(move)` which correctly
  handles the special capture rule.
* Consecutive same-piece moves: we compare the piece on the destination
  square of move N with the source square of move N+1 for the same side.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import chess
import chess.pgn
import pandas as pd

# ---------------------------------------------------------------------------
# Piece values for material balance
# ---------------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

# ---------------------------------------------------------------------------
# PGN helpers
# ---------------------------------------------------------------------------

def _moves_string_to_pgn(moves_string: str) -> str:
    """
    The dataset stores moves as a bare move-text string like
    '1. e4 e5 2. Nf3 Nc6 ... 1-0'.  python-chess expects a full PGN
    file (with headers).  We just prepend a minimal header.
    """
    # Strip trailing result token if present so it doesn't confuse the parser
    cleaned = re.sub(r"\s+(1-0|0-1|1/2-1/2|\*)$", "", moves_string.strip())
    return f"[Event \"?\"]\n\n{cleaned}"


def _parse_game(moves_string: str) -> Optional[chess.pgn.Game]:
    try:
        pgn_text = _moves_string_to_pgn(moves_string)
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        return game
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def extract_features(moves_string: str, result_string: str = "*") -> dict:
    """
    Parse *moves_string* and replay the game to extract 20 hand-crafted
    features.

    Parameters
    ----------
    moves_string  : str
        Raw moves as stored in the Lichess PGN dataset, e.g.
        '1. e4 e5 2. Nf3 Nc6 ... 1-0'
    result_string : str
        The Result tag value from the PGN header ('1-0', '0-1',
        '1/2-1/2', or '*').

    Returns
    -------
    dict of feature_name -> value
    """
    game = _parse_game(moves_string)

    # Fall back to all-zero features if parsing fails
    if game is None:
        return _empty_features()

    board = game.board()
    moves = list(game.mainline_moves())
    total_ply = len(moves)

    # --- per-ply accumulators ---
    checks_white = 0
    checks_black = 0

    first_capture_white = 0   # move number (1-based full moves)
    first_capture_black = 0

    pawn_captures = 0
    piece_captures = 0

    castle_move_white = 0     # full-move number, 0 = never
    castle_move_black = 0
    castle_side_white = "none"
    castle_side_black = "none"

    consec_same_piece_white = 0
    consec_same_piece_black = 0

    queen_moves_before_10 = 0

    # Track destination square of last move per side for consecutive check
    last_dest_white: Optional[chess.Square] = None
    last_dest_black: Optional[chess.Square] = None

    # Max / min rank reached (0-indexed, so rank 0 = rank 1, rank 7 = rank 8)
    max_rank_white = 0   # highest rank index White piece entered (≥ 4 = enemy half)
    min_rank_black = 7   # lowest rank index Black piece entered (≤ 3 = enemy half)

    promotions = 0
    en_passant_captures = 0

    legal_moves_white_move5 = 0
    legal_moves_black_move5 = 0

    for ply_index, move in enumerate(moves):
        full_move_number = board.fullmove_number  # current full-move before push
        is_white_turn = board.turn == chess.WHITE

        # --- legal moves on move 5 (capture BEFORE pushing the move) ---
        if full_move_number == 5:
            if is_white_turn and legal_moves_white_move5 == 0:
                legal_moves_white_move5 = board.legal_moves.count()
            elif not is_white_turn and legal_moves_black_move5 == 0:
                legal_moves_black_move5 = board.legal_moves.count()

        # --- en passant ---
        if board.is_en_passant(move):
            en_passant_captures += 1

        # --- capture details ---
        is_capture = board.is_capture(move)
        if is_capture:
            captured_piece = board.piece_at(move.to_square)
            # For en passant the captured pawn is not on to_square, treat as pawn capture
            if captured_piece is None or captured_piece.piece_type == chess.PAWN:
                pawn_captures += 1
            else:
                piece_captures += 1

            if is_white_turn and first_capture_white == 0:
                first_capture_white = full_move_number
            if not is_white_turn and first_capture_black == 0:
                first_capture_black = full_move_number

        # --- castling ---
        if board.is_castling(move):
            if is_white_turn:
                castle_move_white = full_move_number
                castle_side_white = "king" if board.is_kingside_castling(move) else "queen"
            else:
                castle_move_black = full_move_number
                castle_side_black = "king" if board.is_kingside_castling(move) else "queen"

        # --- promotion ---
        if move.promotion is not None:
            promotions += 1

        # --- territory depth (rank index of destination) ---
        dest_rank = chess.square_rank(move.to_square)  # 0 = rank 1, 7 = rank 8
        if is_white_turn:
            if dest_rank > max_rank_white:
                max_rank_white = dest_rank
        else:
            if dest_rank < min_rank_black:
                min_rank_black = dest_rank

        # --- queen moves before move 10 ---
        if full_move_number < 10:
            moving_piece = board.piece_at(move.from_square)
            if moving_piece and moving_piece.piece_type == chess.QUEEN:
                queen_moves_before_10 += 1

        # --- consecutive same-piece moves ---
        # "Same piece moved consecutively" means side A moves piece P,
        # then on side A's very next turn, moves P again.
        if is_white_turn:
            if last_dest_white is not None and move.from_square == last_dest_white:
                consec_same_piece_white += 1
            last_dest_white = move.to_square
        else:
            if last_dest_black is not None and move.from_square == last_dest_black:
                consec_same_piece_black += 1
            last_dest_black = move.to_square

        # Push the move AFTER all pre-move checks
        board.push(move)

        # --- checks (evaluated AFTER push: is the opponent in check?) ---
        if board.is_check():
            if is_white_turn:   # White just moved → Black is in check
                checks_white += 1
            else:               # Black just moved → White is in check
                checks_black += 1

    # --- material balance at game end ---
    white_material = sum(
        PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE))
        for pt in PIECE_VALUES
    )
    black_material = sum(
        PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK))
        for pt in PIECE_VALUES
    )
    material_balance = white_material - black_material

    # --- territory depth: how many ranks into enemy half? ---
    # White enemy territory starts at rank index 4 (rank 5).
    # Black enemy territory starts at rank index 3 (rank 4).
    white_territory_depth = max(0, max_rank_white - 3)  # 0..4
    black_territory_depth = max(0, 4 - min_rank_black)  # 0..4

    # --- game result encoding ---
    result_map = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "*": 0}
    result_encoded = result_map.get(result_string, 0)

    return {
        # 1
        "total_ply_count": total_ply,
        # 2
        "material_balance_end": material_balance,
        # 3
        "checks_given_white": checks_white,
        # 4
        "checks_given_black": checks_black,
        # 5
        "first_capture_move_white": first_capture_white,
        # 6
        "first_capture_move_black": first_capture_black,
        # 7
        "pawn_captures_total": pawn_captures,
        # 8
        "piece_captures_total": piece_captures,
        # 9
        "castle_move_white": castle_move_white,
        # 10
        "castle_move_black": castle_move_black,
        # 11
        "castle_side_white": castle_side_white,
        # 12
        "castle_side_black": castle_side_black,
        # 13
        "result_encoded": result_encoded,
        # 14
        "consec_same_piece_white": consec_same_piece_white,
        # 15
        "consec_same_piece_black": consec_same_piece_black,
        # 16
        "queen_moves_before_10": queen_moves_before_10,
        # 17
        "white_territory_depth": white_territory_depth,
        # 18
        "black_territory_depth": black_territory_depth,
        # 19
        "promotions": promotions,
        # 20
        "en_passant_captures": en_passant_captures,
        # 21
        "legal_moves_white_move5": legal_moves_white_move5,
        # 22
        "legal_moves_black_move5": legal_moves_black_move5,
    }


def _empty_features() -> dict:
    """Return a zero-filled feature dict for games that fail to parse."""
    return {
        "total_ply_count": 0,
        "material_balance_end": 0,
        "checks_given_white": 0,
        "checks_given_black": 0,
        "first_capture_move_white": 0,
        "first_capture_move_black": 0,
        "pawn_captures_total": 0,
        "piece_captures_total": 0,
        "castle_move_white": 0,
        "castle_move_black": 0,
        "castle_side_white": "none",
        "castle_side_black": "none",
        "result_encoded": 0,
        "consec_same_piece_white": 0,
        "consec_same_piece_black": 0,
        "queen_moves_before_10": 0,
        "white_territory_depth": 0,
        "black_territory_depth": 0,
        "promotions": 0,
        "en_passant_captures": 0,
        "legal_moves_white_move5": 0,
        "legal_moves_black_move5": 0,
    }


# ---------------------------------------------------------------------------
# DataFrame-level helper
# ---------------------------------------------------------------------------

def extract_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply `extract_features` row-wise to a DataFrame that has:
      - 'Moves'  : the raw PGN moves string
      - 'Result' : the PGN result tag  (optional; defaults to '*')

    Returns a new DataFrame of features (same index as *df*).
    """
    result_col = df["Result"] if "Result" in df.columns else pd.Series(["*"] * len(df), index=df.index)

    records = [
        extract_features(moves, result)
        for moves, result in zip(df["Moves"], result_col)
    ]
    return pd.DataFrame(records, index=df.index)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_pgn = (
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 "
        "6. d4 exd4 7. e5 Ne4 8. Nxd4 O-O 9. Nxc6 dxc6 "
        "10. Qxd8 Rxd8 11. Bxc6 bxc6 12. Nxe4 Bb4+ "
        "13. c3 Be7 14. Nd6 Bxd6 15. exd6 Rxd6 1-0"
    )

    feats = extract_features(sample_pgn, "1-0")
    print("Feature extraction smoke-test")
    print("-" * 40)
    for k, v in feats.items():
        print(f"  {k:<35} {v}")
