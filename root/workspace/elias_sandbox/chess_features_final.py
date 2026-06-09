"""
chess_features_final.py
-----------------------
Feature extraction for chess Elo prediction.
Replays each game on a real board via python-chess to compute board-aware
features. Clock features are extracted separately via fast regex in the notebook.

All features are grouped and toggled via FEATURE_GROUPS in the notebook config.
"""

from __future__ import annotations
import io, re
from typing import Optional

import chess
import chess.pgn
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ── piece values ──────────────────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9,  chess.KING: 0,
}

_EVAL_RE = re.compile(r'\[%eval\s+([^\]]+)\]')


def _eval_to_cp(s: str) -> Optional[int]:
    """Eval string → centipawns. Mate scores clamped to ±10000."""
    if not s:
        return None
    if '#' in s:
        try:
            return 10000 if int(s.replace('#', '')) > 0 else -10000
        except ValueError:
            return None
    try:
        return int(float(s) * 100)
    except ValueError:
        return None


def _parse_game(moves_string: str) -> Optional[chess.pgn.Game]:
    """Parse bare move-text (old format) into a Game object."""
    try:
        cleaned = re.sub(r'\s+(1-0|0-1|1/2-1/2|\*)$', '', moves_string.strip())
        return chess.pgn.read_game(io.StringIO(f'[Event "?"]\n\n{cleaned}'))
    except Exception:
        return None


# ── core extraction ───────────────────────────────────────────────────────────

def extract_features(moves_string: str, result_string: str = '*') -> dict:
    """
    Replay one game and return all board-derived features.

    Parameters
    ----------
    moves_string  : raw PGN move text (bare, old Lichess format).
                    Clock features are handled separately in the notebook.
    result_string : '1-0', '0-1', '1/2-1/2', or '*'
    """
    game = _parse_game(moves_string)
    if game is None:
        return _empty_features()

    board = game.board()
    moves = list(game.mainline_moves())

    checks_w = checks_b = 0
    first_cap_w = first_cap_b = 0
    pawn_caps = piece_caps = 0
    castle_move_w = castle_move_b = 0
    castle_side_w = castle_side_b = 'none'
    consec_w = consec_b = 0
    queen_before_10 = 0
    max_rank_w = 0
    min_rank_b = 7
    promotions = en_passant = 0
    legal5_w = legal5_b = 0
    last_dest_w = last_dest_b = None

    # Engine eval (from [%eval] tags if present in the move string)
    cp_losses_w: list[int] = []
    cp_losses_b: list[int] = []
    prev_eval: Optional[int] = 20

    node = game
    while node.variations:
        node = node.variations[0]
        move          = node.move
        comment       = node.comment
        fmn           = board.fullmove_number
        is_white      = board.turn == chess.WHITE

        # legal moves on move 5 (before push)
        if fmn == 5:
            if is_white and legal5_w == 0:
                legal5_w = board.legal_moves.count()
            elif not is_white and legal5_b == 0:
                legal5_b = board.legal_moves.count()

        # engine eval → centipawn loss
        m_eval = _EVAL_RE.search(comment)
        if m_eval:
            cur = _eval_to_cp(m_eval.group(1))
            if cur is not None and prev_eval is not None:
                loss = max(0, min(1000, (prev_eval - cur) if is_white else (cur - prev_eval)))
                (cp_losses_w if is_white else cp_losses_b).append(loss)
            prev_eval = cur
        else:
            prev_eval = None

        # en passant
        if board.is_en_passant(move):
            en_passant += 1

        # captures
        if board.is_capture(move):
            cp = board.piece_at(move.to_square)
            if cp is None or cp.piece_type == chess.PAWN:
                pawn_caps += 1
            else:
                piece_caps += 1
            if is_white and first_cap_w == 0: first_cap_w = fmn
            if not is_white and first_cap_b == 0: first_cap_b = fmn

        # castling
        if board.is_castling(move):
            if is_white:
                castle_move_w = fmn
                castle_side_w = 'king' if board.is_kingside_castling(move) else 'queen'
            else:
                castle_move_b = fmn
                castle_side_b = 'king' if board.is_kingside_castling(move) else 'queen'

        if move.promotion is not None:
            promotions += 1

        dest_rank = chess.square_rank(move.to_square)
        if is_white: max_rank_w = max(max_rank_w, dest_rank)
        else:        min_rank_b = min(min_rank_b, dest_rank)

        if fmn < 10:
            mp = board.piece_at(move.from_square)
            if mp and mp.piece_type == chess.QUEEN:
                queen_before_10 += 1

        if is_white:
            if last_dest_w is not None and move.from_square == last_dest_w: consec_w += 1
            last_dest_w = move.to_square
        else:
            if last_dest_b is not None and move.from_square == last_dest_b: consec_b += 1
            last_dest_b = move.to_square

        board.push(move)

        if board.is_check():
            if is_white: checks_w += 1
            else:        checks_b += 1

    total_ply = len(moves)

    # material
    w_mat = sum(PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE)) for pt in PIECE_VALUES)
    b_mat = sum(PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK)) for pt in PIECE_VALUES)

    # result
    result_enc = {'1-0': 1, '0-1': -1, '1/2-1/2': 0, '*': 0}.get(result_string, 0)

    # engine quality stats
    def _acpl(losses):
        if not losses: return float('nan'), float('nan'), float('nan'), float('nan')
        a = np.array(losses)
        return float(a.mean()), float(np.sum((a>=50)&(a<100))), float(np.sum((a>=100)&(a<300))), float(np.sum(a>=300))

    acpl_w, inacc_w, mist_w, blun_w = _acpl(cp_losses_w)
    acpl_b, inacc_b, mist_b, blun_b = _acpl(cp_losses_b)

    sp = max(1, total_ply)
    w_sp = max(1, total_ply // 2 + total_ply % 2)
    b_sp = max(1, total_ply // 2)

    return {
        # ── game structure ──────────────────────────────────────────────────
        'total_ply_count':            total_ply,
        'material_balance_end':       w_mat - b_mat,
        'result_encoded':             result_enc,
        # ── checks ─────────────────────────────────────────────────────────
        'checks_given_white':         checks_w,
        'checks_given_black':         checks_b,
        'check_density_white':        checks_w / w_sp,
        'check_density_black':        checks_b / b_sp,
        # ── captures ───────────────────────────────────────────────────────
        'first_capture_move_white':   first_cap_w,
        'first_capture_move_black':   first_cap_b,
        'pawn_captures_total':        pawn_caps,
        'piece_captures_total':       piece_caps,
        'capture_density':            (pawn_caps + piece_caps) / sp,
        # ── castling ───────────────────────────────────────────────────────
        'castle_move_white':          castle_move_w,
        'castle_move_black':          castle_move_b,
        'castle_side_white':          castle_side_w,   # categorical
        'castle_side_black':          castle_side_b,   # categorical
        # ── style ──────────────────────────────────────────────────────────
        'consec_same_piece_white':    consec_w,
        'consec_same_piece_black':    consec_b,
        'queen_moves_before_10':      queen_before_10,
        'white_territory_depth':      max(0, max_rank_w - 3),
        'black_territory_depth':      max(0, 4 - min_rank_b),
        'promotions':                 promotions,
        'en_passant_captures':        en_passant,
        'legal_moves_white_move5':    legal5_w,
        'legal_moves_black_move5':    legal5_b,
        # ── engine quality (NaN when evals absent) ─────────────────────────
        'acpl_white':                 acpl_w,
        'inaccuracy_count_white':     inacc_w,
        'mistake_count_white':        mist_w,
        'blunder_count_white':        blun_w,
        'blunder_density_white':      blun_w / w_sp if not np.isnan(blun_w) else float('nan'),
        'acpl_black':                 acpl_b,
        'inaccuracy_count_black':     inacc_b,
        'mistake_count_black':        mist_b,
        'blunder_count_black':        blun_b,
        'blunder_density_black':      blun_b / b_sp if not np.isnan(blun_b) else float('nan'),
    }


def _empty_features() -> dict:
    d = {k: float('nan') for k in [
        'total_ply_count', 'material_balance_end', 'result_encoded',
        'checks_given_white', 'checks_given_black',
        'check_density_white', 'check_density_black',
        'first_capture_move_white', 'first_capture_move_black',
        'pawn_captures_total', 'piece_captures_total', 'capture_density',
        'castle_move_white', 'castle_move_black',
        'consec_same_piece_white', 'consec_same_piece_black',
        'queen_moves_before_10', 'white_territory_depth', 'black_territory_depth',
        'promotions', 'en_passant_captures',
        'legal_moves_white_move5', 'legal_moves_black_move5',
        'acpl_white', 'inaccuracy_count_white', 'mistake_count_white',
        'blunder_count_white', 'blunder_density_white',
        'acpl_black', 'inaccuracy_count_black', 'mistake_count_black',
        'blunder_count_black', 'blunder_density_black',
    ]}
    d['castle_side_white'] = 'none'
    d['castle_side_black'] = 'none'
    return d


# ── parallel DataFrame helper ─────────────────────────────────────────────────

def _extract_row(moves: str, result: str) -> dict:
    return extract_features(moves, result)


def extract_features_dataframe(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Extract board features for every row in parallel.
    Expects columns: 'Moves', optionally 'Result'.
    """
    results_col = df['Result'].tolist() if 'Result' in df.columns else ['*'] * len(df)
    records = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_extract_row)(m, r)
        for m, r in zip(df['Moves'].tolist(), results_col)
    )
    return pd.DataFrame(records, index=df.index)


# ── balanced sample weights ───────────────────────────────────────────────────

def compute_elo_sample_weights(elo_series, bins=None, max_ratio=5.0):
    if bins is None:
        bins = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 4000]
    labels   = pd.cut(elo_series, bins=bins, labels=False, right=True)
    counts   = labels.value_counts()
    inv_freq = 1.0 / counts
    weights  = labels.map(inv_freq).astype(float).fillna(1.0).values
    weights  = weights / weights.mean()
    # cap the maximum ratio so no bracket is oversampled more than max_ratio times
    weights  = np.clip(weights, 0, max_ratio)
    return weights / weights.mean()
