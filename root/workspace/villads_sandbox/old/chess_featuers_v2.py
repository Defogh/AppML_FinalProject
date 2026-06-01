"""
chess_features_v3.py
-----------------
Converts a full PGN game string (Lichess format with [%clk] and [%eval]
annotations) into a rich feature vector by replaying the game on a real
chess board via `python-chess`.

New in v3
---------
* Quality of Play (Engine) features — Extracts Average Centipawn Loss (ACPL), 
  inaccuracies, mistakes, and blunders from [%eval ...] tags if available.
* Density features — capture density, check density, and blunder density to 
  contextualize the raw ply counts.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import chess
import chess.pgn
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

_CLK_RE   = re.compile(r'\[%clk\s+(\d+):(\d+):(\d+)\]')
_TC_RE    = re.compile(r'^(\d+)\+(\d+)$')
_EVAL_RE  = re.compile(r'\[%eval\s+([^\]]+)\]')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clk_to_seconds(comment: str) -> Optional[float]:
    """Extract clock reading (seconds) from a PGN comment, or None."""
    m = _CLK_RE.search(comment)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def _parse_time_control(tc_string: str) -> tuple[float, float]:
    """Return (base_seconds, increment_seconds) or (nan, 0)."""
    m = _TC_RE.match(str(tc_string).strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    return float("nan"), 0.0


def _eval_to_cp(eval_str: str) -> Optional[int]:
    """Convert an eval string (like 0.17, -1.5, or #3) into centipawns."""
    if not eval_str:
        return None
    if '#' in eval_str:
        # Mate score. Positive is winning for White, negative for Black.
        try:
            mate_in = int(eval_str.replace('#', ''))
            return 10000 if mate_in > 0 else -10000
        except ValueError:
            return None
    try:
        return int(float(eval_str) * 100)
    except ValueError:
        return None


def _parse_full_pgn(pgn_string: str) -> Optional[chess.pgn.Game]:
    """
    Parse a complete PGN block (headers + annotated moves).
    Falls back to bare move-text wrapped in a minimal header if needed.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        if game is not None:
            return game
    except Exception:
        pass
    # Fallback: treat the string as bare move text
    try:
        cleaned = re.sub(r'\s+(1-0|0-1|1/2-1/2|\*)$', '', pgn_string.strip())
        game = chess.pgn.read_game(io.StringIO(f'[Event "?"]\n\n{cleaned}'))
        return game
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_features(pgn_string: str,
                     result_string: str = "*",
                     time_control: str = "?") -> dict:
    """
    Replay one game and return a flat feature dict.
    """
    game = _parse_full_pgn(pgn_string)
    if game is None:
        return _empty_features()

    # Prefer header values when available
    hdrs = game.headers
    result_string  = hdrs.get("Result",      result_string)
    time_control   = hdrs.get("TimeControl", time_control)
    _, increment   = _parse_time_control(time_control)

    board = game.board()

    # ── accumulators ────────────────────────────────────────────────────────
    checks_white = checks_black = 0
    first_cap_white = first_cap_black = 0
    pawn_caps = piece_caps = 0
    castle_move_w = castle_move_b = 0
    castle_side_w = castle_side_b = "none"
    consec_w = consec_b = 0
    queen_before_10 = 0
    max_rank_w = 0
    min_rank_b = 7
    promotions = en_passant = 0
    legal5_w = legal5_b = 0

    last_dest_w: Optional[chess.Square] = None
    last_dest_b: Optional[chess.Square] = None

    # Clock tracking
    clk_w: list[float] = []   
    clk_b: list[float] = []   
    time_pressure_w = time_pressure_b = 0   

    # Eval Tracking (Centipawn Loss)
    cp_losses_w: list[int] = []
    cp_losses_b: list[int] = []
    prev_eval: Optional[int] = 20  # Baseline opening advantage 

    node = game
    while node.variations:
        node = node.variations[0]
        move           = node.move
        comment        = node.comment
        full_move_num  = board.fullmove_number
        is_white       = board.turn == chess.WHITE

        # Legal moves on ply 5 (before push)
        if full_move_num == 5:
            if is_white and legal5_w == 0:
                legal5_w = board.legal_moves.count()
            elif not is_white and legal5_b == 0:
                legal5_b = board.legal_moves.count()

        # Clock Processing
        clk_after = _clk_to_seconds(comment)
        if is_white:
            if clk_after is not None:
                clk_w.append(clk_after)
                if clk_after < 10:
                    time_pressure_w += 1
        else:
            if clk_after is not None:
                clk_b.append(clk_after)
                if clk_after < 10:
                    time_pressure_b += 1

        # Evaluation / Centipawn Loss Processing
        m_eval = _EVAL_RE.search(comment)
        if m_eval:
            current_eval = _eval_to_cp(m_eval.group(1))
            if current_eval is not None and prev_eval is not None:
                # Difference relative to the player whose turn it was
                if is_white:
                    cp_loss = prev_eval - current_eval
                else:
                    cp_loss = current_eval - prev_eval
                
                cp_loss = max(0, min(1000, cp_loss)) # Clamp extreme shifts 
                
                if is_white:
                    cp_losses_w.append(cp_loss)
                else:
                    cp_losses_b.append(cp_loss)
            prev_eval = current_eval
        else:
            prev_eval = None # Break chain if evals are missing

        # Movement Trackers
        if board.is_en_passant(move):
            en_passant += 1

        if board.is_capture(move):
            cap_piece = board.piece_at(move.to_square)
            if cap_piece is None or cap_piece.piece_type == chess.PAWN:
                pawn_caps += 1
            else:
                piece_caps += 1
            if is_white and first_cap_white == 0:
                first_cap_white = full_move_num
            if not is_white and first_cap_black == 0:
                first_cap_black = full_move_num

        if board.is_castling(move):
            if is_white:
                castle_move_w = full_move_num
                castle_side_w = "king" if board.is_kingside_castling(move) else "queen"
            else:
                castle_move_b = full_move_num
                castle_side_b = "king" if board.is_kingside_castling(move) else "queen"

        if move.promotion is not None:
            promotions += 1

        dest_rank = chess.square_rank(move.to_square)
        if is_white:
            max_rank_w = max(max_rank_w, dest_rank)
        else:
            min_rank_b = min(min_rank_b, dest_rank)

        if full_move_num < 10:
            mp = board.piece_at(move.from_square)
            if mp and mp.piece_type == chess.QUEEN:
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
                checks_white += 1
            else:
                checks_black += 1

    total_ply = len(clk_w) + len(clk_b)

    # ── clock-derived features ───────────────────────────────────────────────
    def _time_spent(clk_seq: list[float], start_clk: Optional[float]) -> list[float]:
        if not clk_seq:
            return []
        spent = []
        prev  = start_clk if start_clk is not None else None
        for clk in clk_seq:
            if prev is not None:
                s = prev - clk + increment
                if s >= 0:
                    spent.append(s)
            prev = clk
        return spent

    base_sec, _ = _parse_time_control(time_control)
    start = base_sec if not np.isnan(base_sec) else None

    spent_w = _time_spent(clk_w, start)
    spent_b = _time_spent(clk_b, start)

    def _safe_stats(seq: list[float]) -> tuple[float, float, float, float]:
        if not seq:
            return 0.0, 0.0, 0.0, 0.0
        a = np.array(seq)
        return float(a.mean()), float(a.std()), float(a.max()), float(a[-1])

    avg_t_w, std_t_w, max_t_w, last_clk_w = _safe_stats(spent_w)
    avg_t_b, std_t_b, max_t_b, last_clk_b = _safe_stats(spent_b)
    open_pace_w = float(np.mean(spent_w[:10])) if spent_w else 0.0
    open_pace_b = float(np.mean(spent_b[:10])) if spent_b else 0.0
    end_clk_w = clk_w[-1] if clk_w else 0.0
    end_clk_b = clk_b[-1] if clk_b else 0.0

    # ── Engine / Quality Features ────────────────────────────────────────────
    def _acpl_stats(losses: list[int]):
        if not losses:
            return float('nan'), float('nan'), float('nan'), float('nan')
        arr = np.array(losses)
        return float(arr.mean()), float(np.sum((arr >= 50) & (arr < 100))), float(np.sum((arr >= 100) & (arr < 300))), float(np.sum(arr >= 300))

    acpl_w, inacc_w, mist_w, blun_w = _acpl_stats(cp_losses_w)
    acpl_b, inacc_b, mist_b, blun_b = _acpl_stats(cp_losses_b)

    # ── Density Features ─────────────────────────────────────────────────────
    safe_ply = max(1, total_ply)
    w_ply_safe = max(1, len(clk_w))
    b_ply_safe = max(1, len(clk_b))

    capture_density = (pawn_caps + piece_caps) / safe_ply
    check_density_w = checks_white / w_ply_safe
    check_density_b = checks_black / b_ply_safe
    blunder_density_w = blun_w / w_ply_safe if not np.isnan(blun_w) else float('nan')
    blunder_density_b = blun_b / b_ply_safe if not np.isnan(blun_b) else float('nan')

    # ── material balance ─────────────────────────────────────────────────────
    w_mat = sum(PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE)) for pt in PIECE_VALUES)
    b_mat = sum(PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK)) for pt in PIECE_VALUES)
    result_map = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "*": 0}
    result_enc = result_map.get(result_string, 0)

    return {
        "total_ply_count":            total_ply,
        "material_balance_end":       w_mat - b_mat,
        "checks_given_white":         checks_white,
        "checks_given_black":         checks_black,
        "first_capture_move_white":   first_cap_white,
        "first_capture_move_black":   first_cap_black,
        "pawn_captures_total":        pawn_caps,
        "piece_captures_total":       piece_caps,
        "castle_move_white":          castle_move_w,
        "castle_move_black":          castle_move_b,
        "castle_side_white":          castle_side_w,
        "castle_side_black":          castle_side_b,
        "result_encoded":             result_enc,
        "consec_same_piece_white":    consec_w,
        "consec_same_piece_black":    consec_b,
        "queen_moves_before_10":      queen_before_10,
        "white_territory_depth":      max(0, max_rank_w - 3),
        "black_territory_depth":      max(0, 4 - min_rank_b),
        "promotions":                 promotions,
        "en_passant_captures":        en_passant,
        "legal_moves_white_move5":    legal5_w,
        "legal_moves_black_move5":    legal5_b,
        "avg_time_per_move_white":    avg_t_w,
        "std_time_per_move_white":    std_t_w,
        "max_time_single_move_white": max_t_w,
        "time_pressure_moves_white":  time_pressure_w,
        "opening_pace_white":         open_pace_w,
        "clock_remaining_white":      end_clk_w,
        "avg_time_per_move_black":    avg_t_b,
        "std_time_per_move_black":    std_t_b,
        "max_time_single_move_black": max_t_b,
        "time_pressure_moves_black":  time_pressure_b,
        "opening_pace_black":         open_pace_b,
        "clock_remaining_black":      end_clk_b,
        
        # New V3 Features
        "capture_density":            capture_density,
        "check_density_white":        check_density_w,
        "check_density_black":        check_density_b,
        "acpl_white":                 acpl_w,
        "inaccuracy_count_white":     inacc_w,
        "mistake_count_white":        mist_w,
        "blunder_count_white":        blun_w,
        "blunder_density_white":      blunder_density_w,
        "acpl_black":                 acpl_b,
        "inaccuracy_count_black":     inacc_b,
        "mistake_count_black":        mist_b,
        "blunder_count_black":        blun_b,
        "blunder_density_black":      blunder_density_b,
    }


# ---------------------------------------------------------------------------
# Empty features sentinel
# ---------------------------------------------------------------------------

def _empty_features() -> dict:
    keys = [
        "total_ply_count", "material_balance_end",
        "checks_given_white", "checks_given_black",
        "first_capture_move_white", "first_capture_move_black",
        "pawn_captures_total", "piece_captures_total",
        "castle_move_white", "castle_move_black",
        "castle_side_white", "castle_side_black",
        "result_encoded",
        "consec_same_piece_white", "consec_same_piece_black",
        "queen_moves_before_10",
        "white_territory_depth", "black_territory_depth",
        "promotions", "en_passant_captures",
        "legal_moves_white_move5", "legal_moves_black_move5",
        "avg_time_per_move_white", "std_time_per_move_white",
        "max_time_single_move_white", "time_pressure_moves_white",
        "opening_pace_white", "clock_remaining_white",
        "avg_time_per_move_black", "std_time_per_move_black",
        "max_time_single_move_black", "time_pressure_moves_black",
        "opening_pace_black", "clock_remaining_black",
        "capture_density", "check_density_white", "check_density_black",
        "acpl_white", "inaccuracy_count_white", "mistake_count_white", 
        "blunder_count_white", "blunder_density_white",
        "acpl_black", "inaccuracy_count_black", "mistake_count_black", 
        "blunder_count_black", "blunder_density_black",
    ]
    d = {k: float('nan') for k in keys} # Fillna natively using NaNs
    d["castle_side_white"] = "none"
    d["castle_side_black"] = "none"
    return d


# ---------------------------------------------------------------------------
# DataFrame helper — parallel
# ---------------------------------------------------------------------------

def _extract_row(moves: str, result: str, tc: str) -> dict:
    """Thin wrapper so joblib can pickle it."""
    return extract_features(moves, result, tc)


def extract_features_dataframe(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    results_col = df["Result"].tolist()      if "Result"      in df.columns else ["*"]      * len(df)
    tc_col      = df["TimeControl"].tolist() if "TimeControl" in df.columns else ["?"]      * len(df)
    moves_col   = df["Moves"].tolist()

    records = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_extract_row)(m, r, t)
        for m, r, t in zip(moves_col, results_col, tc_col)
    )
    return pd.DataFrame(records, index=df.index)


# ---------------------------------------------------------------------------
# Balanced sampling weights
# ---------------------------------------------------------------------------

def compute_elo_sample_weights(elo_series: pd.Series,
                               bins: list[int] | None = None) -> np.ndarray:
    if bins is None:
        bins = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 4000]

    labels  = pd.cut(elo_series, bins=bins, labels=False, right=True)
    counts  = labels.value_counts()
    inv_freq = 1.0 / counts
    weights  = labels.map(inv_freq).astype(float).fillna(1.0).values
    weights  = weights / weights.mean()
    return weights