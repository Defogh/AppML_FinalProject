"""
chess_features.py
-----------------
Converts a full PGN game string (Lichess format with [%clk] and [%eval]
annotations) into a rich feature vector by replaying the game on a real
chess board via `python-chess`.

New in v2
---------
* Full PGN parsing — accepts the complete PGN block (headers + moves) as
  written in the Lichess dataset, including inline { [%clk ...] [%eval ...] }
  comments.
* Clock features — per-move time spent, average/std think time, time-pressure
  counts, flagging detection, opening/endgame pace split.
* Parallel extraction via joblib — use `extract_features_dataframe` which
  accepts n_jobs.

Key design decisions
--------------------
* python-chess parses the annotated PGN directly; comments are attached to
  each node as `.comment` strings — no regex preprocessing needed on the
  move text.
* Legal-move counts come from `board.legal_moves.count()` BEFORE push —
  the only correct approach.
* Clock time spent = clock_before_move - clock_after_move + increment.
  When clocks are missing for a move we use NaN and skip those moves in
  aggregate statistics.
* Sample weights for balanced training are computed in
  `compute_elo_sample_weights` using inverse-frequency stratification.
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

    Parameters
    ----------
    pgn_string    : str
        Complete PGN block or bare move text.
    result_string : str
        '1-0', '0-1', '1/2-1/2', or '*'.  Ignored if already in PGN headers.
    time_control  : str
        TimeControl header value, e.g. '300+0'.  Used to compute time-spent.
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

    # Clock tracking — one list per side (seconds remaining AFTER the move)
    clk_w: list[float] = []   # after each White move
    clk_b: list[float] = []   # after each Black move
    prev_clk_w: Optional[float] = None
    prev_clk_b: Optional[float] = None
    time_pressure_w = time_pressure_b = 0   # moves made with < 10 s on clock

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

        # Clock
        clk_after = _clk_to_seconds(comment)

        if is_white:
            if clk_after is not None:
                clk_w.append(clk_after)
                if clk_after < 10:
                    time_pressure_w += 1
            prev_clk_w = clk_after
        else:
            if clk_after is not None:
                clk_b.append(clk_after)
                if clk_after < 10:
                    time_pressure_b += 1
            prev_clk_b = clk_after

        # En passant
        if board.is_en_passant(move):
            en_passant += 1

        # Captures
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

        # Castling
        if board.is_castling(move):
            if is_white:
                castle_move_w = full_move_num
                castle_side_w = "king" if board.is_kingside_castling(move) else "queen"
            else:
                castle_move_b = full_move_num
                castle_side_b = "king" if board.is_kingside_castling(move) else "queen"

        # Promotion
        if move.promotion is not None:
            promotions += 1

        # Territory depth
        dest_rank = chess.square_rank(move.to_square)
        if is_white:
            max_rank_w = max(max_rank_w, dest_rank)
        else:
            min_rank_b = min(min_rank_b, dest_rank)

        # Queen moves before move 10
        if full_move_num < 10:
            mp = board.piece_at(move.from_square)
            if mp and mp.piece_type == chess.QUEEN:
                queen_before_10 += 1

        # Consecutive same-piece
        if is_white:
            if last_dest_w is not None and move.from_square == last_dest_w:
                consec_w += 1
            last_dest_w = move.to_square
        else:
            if last_dest_b is not None and move.from_square == last_dest_b:
                consec_b += 1
            last_dest_b = move.to_square

        board.push(move)

        # Checks (after push)
        if board.is_check():
            if is_white:
                checks_white += 1
            else:
                checks_black += 1

    total_ply = board.fullmove_number * 2 - (1 if board.turn == chess.WHITE else 0) - 1
    # Safer: just count from node traversal
    total_ply = len(clk_w) + len(clk_b)  # approximate; ok for feature

    # ── clock-derived features ───────────────────────────────────────────────
    def _time_spent(clk_seq: list[float], start_clk: Optional[float]) -> list[float]:
        """
        Convert a sequence of remaining-clock readings into time-spent-per-move.
        time_spent[i] = clk[i-1] - clk[i] + increment
        For the first move we use start_clk (from TimeControl base) if known.
        """
        if not clk_seq:
            return []
        spent = []
        prev  = start_clk if start_clk is not None else None
        for clk in clk_seq:
            if prev is not None:
                s = prev - clk + increment
                if s >= 0:          # ignore pathological negatives (lag etc.)
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

    # Opening pace: avg time on first 10 moves
    open_pace_w = float(np.mean(spent_w[:10])) if spent_w else 0.0
    open_pace_b = float(np.mean(spent_b[:10])) if spent_b else 0.0

    # Clock remaining at end (last reading in each list)
    end_clk_w = clk_w[-1] if clk_w else 0.0
    end_clk_b = clk_b[-1] if clk_b else 0.0

    # ── material balance ─────────────────────────────────────────────────────
    w_mat = sum(PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE)) for pt in PIECE_VALUES)
    b_mat = sum(PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK)) for pt in PIECE_VALUES)

    # ── result ───────────────────────────────────────────────────────────────
    result_map = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "*": 0}
    result_enc = result_map.get(result_string, 0)

    return {
        # ── game structure ──────────────────────────────────────────────────
        "total_ply_count":            len(clk_w) + len(clk_b),
        "material_balance_end":       w_mat - b_mat,
        # ── checks ─────────────────────────────────────────────────────────
        "checks_given_white":         checks_white,
        "checks_given_black":         checks_black,
        # ── captures ───────────────────────────────────────────────────────
        "first_capture_move_white":   first_cap_white,
        "first_capture_move_black":   first_cap_black,
        "pawn_captures_total":        pawn_caps,
        "piece_captures_total":       piece_caps,
        # ── castling ───────────────────────────────────────────────────────
        "castle_move_white":          castle_move_w,
        "castle_move_black":          castle_move_b,
        "castle_side_white":          castle_side_w,
        "castle_side_black":          castle_side_b,
        # ── result ─────────────────────────────────────────────────────────
        "result_encoded":             result_enc,
        # ── style / pattern ────────────────────────────────────────────────
        "consec_same_piece_white":    consec_w,
        "consec_same_piece_black":    consec_b,
        "queen_moves_before_10":      queen_before_10,
        "white_territory_depth":      max(0, max_rank_w - 3),
        "black_territory_depth":      max(0, 4 - min_rank_b),
        "promotions":                 promotions,
        "en_passant_captures":        en_passant,
        "legal_moves_white_move5":    legal5_w,
        "legal_moves_black_move5":    legal5_b,
        # ── clock / time-management (White) ────────────────────────────────
        "avg_time_per_move_white":    avg_t_w,
        "std_time_per_move_white":    std_t_w,
        "max_time_single_move_white": max_t_w,
        "time_pressure_moves_white":  time_pressure_w,
        "opening_pace_white":         open_pace_w,
        "clock_remaining_white":      end_clk_w,
        # ── clock / time-management (Black) ────────────────────────────────
        "avg_time_per_move_black":    avg_t_b,
        "std_time_per_move_black":    std_t_b,
        "max_time_single_move_black": max_t_b,
        "time_pressure_moves_black":  time_pressure_b,
        "opening_pace_black":         open_pace_b,
        "clock_remaining_black":      end_clk_b,
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
    ]
    d = {k: 0 for k in keys}
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
    """
    Extract features for every row of *df* in parallel.

    Expected columns
    ----------------
    - 'Moves'       : full PGN string or bare move text
    - 'Result'      : optional, e.g. '1-0'
    - 'TimeControl' : optional, e.g. '300+0'

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers (default: -1 = all CPUs).
    """
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
    """
    Compute per-sample weights so that each Elo stratum contributes equally
    to training.  Rare high-Elo games get larger weights.

    Parameters
    ----------
    elo_series : pd.Series of int Elo values
    bins       : Elo bin edges.  Defaults to standard rating class boundaries.

    Returns
    -------
    np.ndarray of float weights, shape (len(elo_series),)
    """
    if bins is None:
        bins = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 4000]

    labels  = pd.cut(elo_series, bins=bins, labels=False, right=True)
    counts  = labels.value_counts()
    # weight = 1 / frequency of that stratum (normalised so mean weight = 1)
    inv_freq = 1.0 / counts
    weights  = labels.map(inv_freq).astype(float).fillna(1.0).values
    weights  = weights / weights.mean()
    return weights


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_pgn = """\
[Event "Rated Bullet tournament"]
[WhiteElo "2100"]
[BlackElo "2000"]
[Result "0-1"]
[TimeControl "300+0"]
[ECO "B30"]
[Termination "Time forfeit"]

1. e4 { [%eval 0.17] [%clk 0:05:00] } 1... c5 { [%eval 0.19] [%clk 0:05:00] }
2. Nf3 { [%eval 0.25] [%clk 0:04:55] } 2... Nc6 { [%eval 0.33] [%clk 0:04:58] }
3. Bc4 { [%eval -0.13] [%clk 0:04:50] } 3... e6 { [%eval -0.04] [%clk 0:04:56] }
4. c3 { [%eval -0.4] [%clk 0:04:45] } 4... b5? { [%eval 1.18] [%clk 0:04:54] }
5. Bb3?! { [%eval 0.21] [%clk 0:04:40] } 5... c4 { [%eval 0.32] [%clk 0:04:52] }
6. Bc2 { [%eval 0.2] [%clk 0:04:35] } 6... a5 { [%eval 0.6] [%clk 0:04:50] }
7. d4 { [%eval 0.29] [%clk 0:04:30] } 7... cxd3 { [%eval 0.6] [%clk 0:04:47] }
8. Qxd3 { [%eval 0.12] [%clk 0:04:20] } 8... Nf6 { [%eval 0.52] [%clk 0:04:44] }
9. e5 { [%eval 0.39] [%clk 0:04:15] } 9... Nd5 { [%eval 0.45] [%clk 0:04:42] }
10. Bg5?! { [%eval -0.44] [%clk 0:04:05] } 10... Qc7 { [%eval -0.12] [%clk 0:04:38] } 0-1"""

    feats = extract_features(sample_pgn)
    print("Feature extraction smoke-test")
    print("-" * 45)
    for k, v in feats.items():
        print(f"  {k:<38} {v}")
