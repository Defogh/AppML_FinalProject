#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import chess
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "lichess-2017-05-eval-all"
OUT_DIR = ROOT / "misc" 
GAME_INDEX = 5283594

BOARD_SIZE = 640
PANEL_W = 560
W = BOARD_SIZE + PANEL_W
H = 720
SQ = BOARD_SIZE // 8

LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
LAST_MOVE = (245, 211, 92)
CHECK = (205, 69, 69)
BG = (248, 249, 250)
INK = (33, 37, 41)
MUTED = (96, 108, 118)
WHITE_LINE = (39, 119, 219)
BLACK_LINE = (210, 74, 67)
GREEN = (32, 145, 92)
RED = (200, 57, 54)

PIECE_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/DejaVu Sans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


FONT_TITLE = font(34, bold=True)
FONT_BIG = font(28, bold=True)
FONT_MED = font(22)
FONT_SMALL = font(17)
FONT_TINY = font(14)
FONT_PIECE = font(54)


def board_xy(square: chess.Square) -> tuple[int, int]:
    file_i = chess.square_file(square)
    rank_i = chess.square_rank(square)
    return file_i * SQ, (7 - rank_i) * SQ


def cp_to_win_prob(eval_pawns: float) -> float:
    # Smooth display-only transform. Positive eval favors White.
    return 1.0 / (1.0 + math.exp(-eval_pawns / 2.2))


def loss_for_row(row: pd.Series) -> float:
    if row["side"] == "white":
        value = row.get("white_loss_proxy")
    else:
        value = row.get("black_loss_proxy")
    if pd.isna(value):
        return 0.0
    return max(0.0, float(value))


def load_game() -> tuple[pd.Series, pd.DataFrame]:
    game = pd.read_parquet(
        DATA_DIR / "games.parquet", filters=[("game_index", "=", GAME_INDEX)]
    ).iloc[0]
    plies = pd.read_parquet(
        DATA_DIR / "plies.parquet", filters=[("game_index", "=", GAME_INDEX)]
    ).sort_values("ply")
    return game, plies.reset_index(drop=True)


def positions_from_plies(plies: pd.DataFrame) -> tuple[list[chess.Board], list[chess.Move]]:
    board = chess.Board()
    boards = [board.copy()]
    moves: list[chess.Move] = []
    for san in plies["san_raw"]:
        clean_san = re.sub(r"[!?]+$", "", str(san))
        move = board.parse_san(clean_san)
        board.push(move)
        moves.append(move)
        boards.append(board.copy())
    return boards, moves


def draw_board(draw: ImageDraw.ImageDraw, board: chess.Board, last_move: chess.Move | None) -> None:
    last_squares = set()
    if last_move is not None:
        last_squares = {last_move.from_square, last_move.to_square}

    for rank in range(8):
        for file_i in range(8):
            square = chess.square(file_i, 7 - rank)
            x, y = file_i * SQ, rank * SQ
            color = LIGHT if (rank + file_i) % 2 == 0 else DARK
            if square in last_squares:
                color = LAST_MOVE
            draw.rectangle([x, y, x + SQ, y + SQ], fill=color)

    if board.is_check():
        king_square = board.king(board.turn)
        if king_square is not None:
            x, y = board_xy(king_square)
            draw.rectangle([x + 5, y + 5, x + SQ - 5, y + SQ - 5], outline=CHECK, width=5)

    for square, piece in board.piece_map().items():
        x, y = board_xy(square)
        glyph = PIECE_UNICODE[piece.symbol()]
        fill = (245, 245, 245) if piece.color == chess.WHITE else (24, 24, 24)
        stroke = (20, 20, 20) if piece.color == chess.WHITE else (240, 240, 240)
        bbox = draw.textbbox((0, 0), glyph, font=FONT_PIECE, stroke_width=1)
        tx = x + (SQ - (bbox[2] - bbox[0])) / 2
        ty = y + (SQ - (bbox[3] - bbox[1])) / 2 - 4
        draw.text((tx, ty), glyph, font=FONT_PIECE, fill=fill, stroke_width=1, stroke_fill=stroke)

    for i, file_name in enumerate("abcdefgh"):
        draw.text((i * SQ + 6, BOARD_SIZE - 20), file_name, font=FONT_TINY, fill=(55, 65, 70))
    for i in range(8):
        draw.text((5, i * SQ + 4), str(8 - i), font=FONT_TINY, fill=(55, 65, 70))


def draw_eval_chart(
    draw: ImageDraw.ImageDraw,
    plies: pd.DataFrame,
    current_ply: int,
    blunder_ply: int,
    x0: int,
    y0: int,
    w: int,
    h: int,
) -> None:
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=6, fill=(255, 255, 255), outline=(215, 222, 228))
    draw.text((x0 + 18, y0 + 12), "Engine eval swing", font=FONT_MED, fill=INK)
    chart = [x0 + 34, y0 + 56, x0 + w - 22, y0 + h - 34]
    cx0, cy0, cx1, cy1 = chart
    draw.line([cx0, (cy0 + cy1) / 2, cx1, (cy0 + cy1) / 2], fill=(210, 215, 220), width=2)
    draw.text((cx0 - 26, cy0 - 8), "+10", font=FONT_TINY, fill=MUTED)
    draw.text((cx0 - 18, (cy0 + cy1) / 2 - 8), "0", font=FONT_TINY, fill=MUTED)
    draw.text((cx0 - 24, cy1 - 12), "-10", font=FONT_TINY, fill=MUTED)

    evals = plies["eval_proxy"].astype(float).clip(-10, 10).tolist()
    points = []
    total = max(1, len(evals) - 1)
    for i, value in enumerate(evals):
        x = cx0 + (cx1 - cx0) * i / total
        y = cy1 - (cy1 - cy0) * ((value + 10) / 20)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(42, 92, 152), width=3)

    visible_idx = max(0, min(current_ply - 1, len(points) - 1))
    if points:
        x, y = points[visible_idx]
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=GREEN, outline=(255, 255, 255), width=2)
    if 1 <= blunder_ply <= len(points):
        bx, by = points[blunder_ply - 1]
        draw.line([bx, cy0, bx, cy1], fill=RED, width=3)
        draw.text((bx + 8, cy0 + 4), "blunder", font=FONT_TINY, fill=RED)


def draw_win_bar(draw: ImageDraw.ImageDraw, eval_now: float, x: int, y: int, w: int, h: int) -> None:
    p = cp_to_win_prob(eval_now)
    white_w = int(w * p)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=5, fill=BLACK_LINE)
    draw.rounded_rectangle([x, y, x + white_w, y + h], radius=5, fill=WHITE_LINE)
    draw.text((x, y - 25), "Win probability proxy", font=FONT_SMALL, fill=MUTED)
    draw.text((x + 8, y + 6), f"White {p * 100:.0f}%", font=FONT_SMALL, fill=(255, 255, 255))
    black_label = f"Black {(1 - p) * 100:.0f}%"
    tw = draw.textlength(black_label, font=FONT_SMALL)
    draw.text((x + w - tw - 8, y + 6), black_label, font=FONT_SMALL, fill=(255, 255, 255))


def draw_panel(
    img: Image.Image,
    game: pd.Series,
    plies: pd.DataFrame,
    current_ply: int,
    blunder_ply: int,
    blunder_loss: float,
) -> None:
    draw = ImageDraw.Draw(img)
    x = BOARD_SIZE + 34
    row = plies.iloc[current_ply - 1] if current_ply > 0 else None
    eval_now = 0.0 if row is None else float(row["eval_proxy"])
    side = "" if row is None else str(row["side"]).capitalize()
    san = "Start position" if row is None else str(row["san_raw"])
    move_no = "" if row is None else f"{int(row['move_number'])}{'.' if row['side'] == 'white' else '...'}"
    loss = 0.0 if row is None else loss_for_row(row)

    draw.text((x, 38), "Blunder Detection", font=FONT_TITLE, fill=INK)
    draw.text((x, 84), f"Game {int(game['game_index'])} · {game['opening']}", font=FONT_SMALL, fill=MUTED)

    draw.rounded_rectangle([x, 124, x + 492, 234], radius=6, fill=(255, 255, 255), outline=(215, 222, 228))
    draw.text((x + 18, 142), f"White: {game['white']} ({int(game['white_elo'])})", font=FONT_MED, fill=INK)
    draw.text((x + 18, 174), f"Black: {game['black']} ({int(game['black_elo'])})", font=FONT_MED, fill=INK)
    draw.text((x + 18, 204), f"Result: {game['result']} · {game['time_control']}", font=FONT_SMALL, fill=MUTED)

    draw.text((x, 268), f"Move {move_no} {san}", font=FONT_BIG, fill=INK)
    draw.text((x, 306), f"{side} to engine eval: {eval_now:+.2f} pawns", font=FONT_MED, fill=MUTED)
    if current_ply == blunder_ply:
        draw.rounded_rectangle([x, 348, x + 492, 418], radius=6, fill=(255, 236, 235), outline=RED, width=3)
        draw.text((x + 18, 364), f"BLUNDER: {blunder_loss:.2f} pawn loss", font=FONT_BIG, fill=RED)
        draw.text((x + 18, 397), "Model target: sharp eval drop for mover", font=FONT_SMALL, fill=INK)
    else:
        draw.rounded_rectangle([x, 348, x + 492, 418], radius=6, fill=(255, 255, 255), outline=(215, 222, 228))
        draw.text((x + 18, 364), f"Current loss proxy: {loss:.2f}", font=FONT_BIG, fill=INK)
        draw.text((x + 18, 397), f"Blunder marker at ply {blunder_ply}", font=FONT_SMALL, fill=MUTED)

    draw_win_bar(draw, eval_now, x, 466, 492, 40)
    draw_eval_chart(draw, plies, max(current_ply, 1), blunder_ply, x, 548, 492, 142)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    game, plies = load_game()
    boards, moves = positions_from_plies(plies)

    losses = plies.apply(loss_for_row, axis=1)
    blunder_idx = int(losses.idxmax())
    blunder_ply = int(plies.loc[blunder_idx, "ply"])
    blunder_loss = float(losses.loc[blunder_idx])

    frames: list[Image.Image] = []
    for ply in range(0, len(plies) + 1):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        last_move = None if ply == 0 else moves[ply - 1]
        draw_board(draw, boards[ply], last_move)
        draw_panel(img, game, plies, ply, blunder_ply, blunder_loss)
        frames.append(img)
        if ply == blunder_ply:
            frames.extend([img.copy() for _ in range(5)])

    out = OUT_DIR / f"blunder_detection_game_{GAME_INDEX}.gif"
    durations = [260] * len(frames)
    for i in range(max(0, blunder_ply - 1), min(len(durations), blunder_ply + 6)):
        durations[i] = 850
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(out)
    print(f"game_index={GAME_INDEX} blunder_ply={blunder_ply} blunder_loss={blunder_loss:.2f}")


if __name__ == "__main__":
    main()
