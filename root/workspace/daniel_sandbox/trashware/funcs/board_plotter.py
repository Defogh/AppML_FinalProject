from __future__ import annotations

import io
from dataclasses import dataclass

import chess
import chess.pgn
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import ipywidgets as widgets

from IPython.display import display, clear_output


PIECE_TO_UNICODE = {
    "P": "♟", "N": "♞", "B": "♝", "R": "♜", "Q": "♛", "K": "♚",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


BOARD_COLORS = {
    "light_square": "#4A4A50",
    "dark_square": "#2B2B31",
    "highlight": "#8B5CF6",

    "white_piece": "#F8FAFC",
    "black_piece": "#030712",

    "white_piece_outline": "#030712",
    "black_piece_outline": "#F8FAFC",

    "text": "#F3F4F6",
    "background": "#09090B",
}

@dataclass
class ParsedGame:
    boards: list[chess.Board]
    moves: list[chess.Move]
    san_moves: list[str]
    result: str | None


def parse_san_game(san_text: str) -> ParsedGame:
    """
    Parse a SAN/PGN movetext string.

    The board list includes the starting position:

        boards[0] = initial board
        boards[1] = after first half-move
        boards[2] = after second half-move

    Thus, move_index counts half-moves.
    """
    game = chess.pgn.read_game(io.StringIO(san_text))

    if game is None:
        raise ValueError("Could not parse SAN/PGN text.")

    board = game.board()

    boards = [board.copy(stack=False)]
    moves = []
    san_moves = []

    for move in game.mainline_moves():
        san = board.san(move)
        board.push(move)

        moves.append(move)
        san_moves.append(san)
        boards.append(board.copy(stack=False))

    result = game.headers.get("Result")
    if result == "*":
        result = None

    return ParsedGame(
        boards=boards,
        moves=moves,
        san_moves=san_moves,
        result=result,
    )


def _square_to_xy(square: chess.Square, *, flipped: bool = False) -> tuple[int, int]:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)

    if flipped:
        return 7 - file_idx, 7 - rank_idx

    return file_idx, rank_idx


def _move_title(parsed: ParsedGame, move_index: int) -> str:
    if move_index == 0:
        return "Starting position"

    san = parsed.san_moves[move_index - 1]
    fullmove = (move_index + 1) // 2
    side = "White" if move_index % 2 == 1 else "Black"

    return f"After half-move {move_index}: {side} played {san} | full move {fullmove}"


def _draw_board(
    board: chess.Board,
    *,
    ax,
    flipped: bool = False,
    title: str | None = None,
    last_move: chess.Move | None = None,
    colors: dict[str, str] = BOARD_COLORS,
) -> None:
    """
    Internal helper for drawing a chess.Board onto a Matplotlib axis.
    """
    ax.clear()
    ax.set_facecolor(colors["background"])

    # Draw board squares.
    # a1 is dark, so display coordinate (0, 0) should be dark.
    for x in range(8):
        for y in range(8):
            is_light = (x + y) % 2 == 1
            square_color = colors["light_square"] if is_light else colors["dark_square"]

            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=square_color,
                    edgecolor="none",
                )
            )

    # Highlight the previous move.
    if last_move is not None:
        for square in (last_move.from_square, last_move.to_square):
            x, y = _square_to_xy(square, flipped=flipped)

            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=colors["highlight"],
                    edgecolor="none",
                    alpha=0.55,
                )
            )

    # Draw pieces.
    for square, piece in board.piece_map().items():
        x, y = _square_to_xy(square, flipped=flipped)
        symbol = PIECE_TO_UNICODE[piece.symbol()]

        if piece.color == chess.WHITE:
            piece_color = colors["white_piece"]
            outline_color = colors["white_piece_outline"]
            stroke_width = 1.2
        else:
            piece_color = colors["black_piece"]
            outline_color = colors["black_piece_outline"]
            stroke_width = 1.6

        text = ax.text(
            x + 0.5,
            y + 0.5,
            symbol,
            ha="center",
            va="center",
            fontsize=36,
            color=piece_color,
        )

        text.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=stroke_width,
                    foreground=outline_color,
                ),
                path_effects.Normal(),
            ]
        )

    # Coordinates.
    files = list("abcdefgh")
    ranks = list("12345678")

    if flipped:
        files = files[::-1]
        ranks = ranks[::-1]

    ax.set_xticks([i + 0.5 for i in range(8)])
    ax.set_yticks([i + 0.5 for i in range(8)])
    ax.set_xticklabels(files, color=colors["text"])
    ax.set_yticklabels(ranks, color=colors["text"])

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is not None:
        ax.set_title(title, color=colors["text"], pad=12)


def plot_board_single(
    san_text: str,
    move_index: int,
    *,
    flipped: bool = False,
    figsize: tuple[float, float] = (6, 6),
) -> None:
    """
    Plot one static board position from a SAN/PGN movetext string.

    Parameters
    ----------
    san_text:
        SAN/PGN movetext string.

    move_index:
        Number of half-moves to apply before plotting.

        Examples:
            0 -> starting position
            1 -> after White's first move
            2 -> after Black's first move

    flipped:
        If True, show the board from Black's perspective.

    figsize:
        Matplotlib figure size.
    """
    parsed = parse_san_game(san_text)

    if not 0 <= move_index < len(parsed.boards):
        raise IndexError(
            f"move_index must be between 0 and {len(parsed.boards) - 1}."
        )

    board = parsed.boards[move_index]
    last_move = parsed.moves[move_index - 1] if move_index > 0 else None
    title = _move_title(parsed, move_index)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BOARD_COLORS["background"])

    _draw_board(
        board,
        ax=ax,
        flipped=flipped,
        title=title,
        last_move=last_move,
    )

    plt.show()


class _NotebookChessViewer:
    """
    Internal notebook widget viewer.
    """

    def __init__(self, san_text: str, *, flipped: bool = False):
        self.parsed = parse_san_game(san_text)
        self.flipped = flipped
        self.move_index = 0

        max_index = len(self.parsed.boards) - 1

        self.output = widgets.Output()

        self.slider = widgets.IntSlider(
            value=0,
            min=0,
            max=max_index,
            step=1,
            description="Move:",
            continuous_update=False,
            layout=widgets.Layout(width="620px"),
        )

        self.prev_button = widgets.Button(description="Previous")
        self.next_button = widgets.Button(description="Next")
        self.move_label = widgets.HTML()

        self.slider.observe(self._on_slider_changed, names="value")
        self.prev_button.on_click(self._on_prev_clicked)
        self.next_button.on_click(self._on_next_clicked)

        controls = widgets.VBox(
            [
                self.move_label,
                self.slider,
                widgets.HBox([self.prev_button, self.next_button]),
            ]
        )

        display(widgets.VBox([controls, self.output]))

        self._draw()

    def _draw(self) -> None:
        board = self.parsed.boards[self.move_index]
        last_move = (
            self.parsed.moves[self.move_index - 1]
            if self.move_index > 0
            else None
        )

        title = _move_title(self.parsed, self.move_index)
        self.move_label.value = f"<b>{title}</b>"

        with self.output:
            clear_output(wait=True)

            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor(BOARD_COLORS["background"])

            _draw_board(
                board,
                ax=ax,
                flipped=self.flipped,
                title=title,
                last_move=last_move,
            )

            plt.show()
            plt.close(fig)

    def _set_move_index(self, move_index: int) -> None:
        move_index = max(0, min(move_index, len(self.parsed.boards) - 1))
        self.move_index = move_index

        if self.slider.value != move_index:
            self.slider.value = move_index
        else:
            self._draw()

    def _on_slider_changed(self, change) -> None:
        self.move_index = change["new"]
        self._draw()

    def _on_prev_clicked(self, button) -> None:
        self._set_move_index(self.move_index - 1)

    def _on_next_clicked(self, button) -> None:
        self._set_move_index(self.move_index + 1)


def plot_board_full(
    san_text: str,
    *,
    flipped: bool = False,
) -> _NotebookChessViewer:
    """
    Show a notebook-interactive chess viewer with a move slider and buttons.

    Use in a Jupyter notebook:

        viewer = plot_board_full(san)

    Keep the returned viewer assigned to a variable.
    """
    return _NotebookChessViewer(san_text, flipped=flipped)