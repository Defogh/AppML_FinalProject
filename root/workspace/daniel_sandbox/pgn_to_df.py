from pathlib import Path
import chess.pgn
import pandas as pd


def parse_int(value):
    """Convert PGN integer-like values to ``int``; otherwise return ``None``."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parser(pgn_path):
    """Parse a PGN file into a dataframe, skipping games with unknown players."""
    pgn_path = Path(pgn_path)
    games = []

    with pgn_path.open(encoding="utf-8") as file:
        while True:
            game = chess.pgn.read_game(file)

            if game is None:
                break

            headers = dict(game.headers)

            white = headers.get("White")
            black = headers.get("Black")
            white_elo = parse_int(headers.get("WhiteElo"))
            black_elo = parse_int(headers.get("BlackElo"))

            if white in (None, "?") or black in (None, "?"):
                continue

            if white_elo is None or black_elo is None:
                continue

            moves = list(game.mainline_moves())

            games.append(
                {
                    "event": headers.get("Event"),
                    "site": headers.get("Site"),
                    "white": white,
                    "black": black,
                    "result": headers.get("Result"),
                    "date": headers.get("UTCDate"),
                    "time": headers.get("UTCTime"),
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "white_rating_diff": parse_int(headers.get("WhiteRatingDiff")),
                    "black_rating_diff": parse_int(headers.get("BlackRatingDiff")),
                    "eco": headers.get("ECO"),
                    "opening": headers.get("Opening"),
                    "time_control": headers.get("TimeControl"),
                    "termination": headers.get("Termination"),
                    "num_halfmoves": len(moves),
                    "num_moves": len(moves) / 2,
                }
            )

    return pd.DataFrame(games)