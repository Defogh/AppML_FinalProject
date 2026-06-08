from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import bz2
import gzip
import io
import re
from typing import TextIO

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is optional
    tqdm = None

HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$')
BRACE_COMMENT_RE = re.compile(r"\{([^{}]*)\}")
SEMICOLON_COMMENT_RE = re.compile(r";[^\n\r]*")
MOVE_NUMBER_PREFIX_RE = re.compile(r"^\d+\.(?:\.\.?)?")
CLOCK_RE = re.compile(r"\[%clkc?\s+([^\]\s]+)\s*\]")
EVAL_RE = re.compile(r"\[%eval\s+([^\]\s]+)\s*\]")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
LIST_SEP = "|"


def parse_int(value: object) -> int | None:
    try:
        return int(str(value).replace("+", ""))
    except (TypeError, ValueError):
        return None


def parse_result_score(result: str | None) -> float | None:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)


def parse_time_control(time_control: str | None) -> tuple[int | None, int | None]:
    if not time_control or time_control == "-" or "+" not in str(time_control):
        return None, None
    initial, increment = str(time_control).split("+", 1)
    return parse_int(initial), parse_int(increment)


def classify_time_control(initial_seconds: int | None, increment_seconds: int | None) -> str | None:
    if initial_seconds is None or increment_seconds is None:
        return None
    estimated = initial_seconds + 40 * increment_seconds
    if estimated < 180:
        return "bullet"
    if estimated < 480:
        return "blitz"
    if estimated < 1500:
        return "rapid"
    return "classical"


def parse_clock_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    parts = str(value).strip().split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return 3600 * h + 60 * m + s
        if len(parts) == 2:
            m, s = map(float, parts)
            return 60 * m + s
        if len(parts) == 1 and parts[0]:
            return float(parts[0])
    except ValueError:
        return None
    return None


def parse_eval_token(value: str | None) -> tuple[float | None, int | None]:
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
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def serialize_list(values: Sequence[object]) -> str:
    return LIST_SEP.join(_format_scalar(v) for v in values)


def extract_lichess_id(site: str | None) -> str | None:
    if not site:
        return None
    return site.rstrip("/").split("/")[-1] or None


def open_text_maybe_compressed(path: str | Path) -> TextIO:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    if suffix == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    if suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError("Install zstandard to read .pgn.zst files: pip install zstandard") from exc
        stream = zstd.ZstdDecompressor().stream_reader(path.open("rb"))
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def iter_pgn_games(pgn_path: str | Path) -> Iterator[tuple[dict[str, str], str]]:
    headers: dict[str, str] = {}
    movetext_parts: list[str] = []

    with open_text_maybe_compressed(pgn_path) as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                if headers and movetext_parts:
                    yield headers, "\n".join(movetext_parts)
                    headers, movetext_parts = {}, []
                continue

            match = HEADER_RE.match(line)
            if match:
                key, value = match.groups()
                if movetext_parts:
                    yield headers, "\n".join(movetext_parts)
                    headers, movetext_parts = {}, []
                headers[key] = value
            elif headers:
                movetext_parts.append(line)

    if headers and movetext_parts:
        yield headers, "\n".join(movetext_parts)


def _remove_variations(text: str) -> str:
    while "(" in text and ")" in text:
        new = re.sub(r"\([^()]*\)", " ", text)
        if new == text:
            break
        text = new
    return text


def _normalize_san_token(raw_token: str) -> str | None:
    token = MOVE_NUMBER_PREFIX_RE.sub("", raw_token.strip())
    if not token or token == "..." or token in RESULT_TOKENS or token.startswith("$"):
        return None
    token = token.rstrip("!?")
    return token or None


def parse_movetext(movetext: str) -> tuple[list[str], list[float | None], list[float | None], list[int | None]]:
    text = _remove_variations(SEMICOLON_COMMENT_RE.sub(" ", movetext))
    san: list[str] = []
    clocks: list[float | None] = []
    evals: list[float | None] = []
    mates: list[int | None] = []

    def add_tokens(segment: str) -> None:
        for raw_token in segment.split():
            token = _normalize_san_token(raw_token)
            if token is not None:
                san.append(token)
                clocks.append(None)
                evals.append(None)
                mates.append(None)

    pos = 0
    for comment_match in BRACE_COMMENT_RE.finditer(text):
        add_tokens(text[pos:comment_match.start()])
        if san:
            comment = comment_match.group(1)
            clock_match = CLOCK_RE.search(comment)
            eval_match = EVAL_RE.search(comment)
            if clock_match:
                clocks[-1] = parse_clock_seconds(clock_match.group(1))
            if eval_match:
                evals[-1], mates[-1] = parse_eval_token(eval_match.group(1))
        pos = comment_match.end()

    add_tokens(text[pos:])
    return san, clocks, evals, mates


def san_tokens(movetext: str) -> list[str]:
    return parse_movetext(movetext)[0]


def game_to_row(
    headers: dict[str, str],
    movetext: str,
    *,
    game_index: int | None = None,
    include_moves_san: bool = True,
    include_annotation_series: bool = True,
    include_moves_pgn: bool = False,
) -> dict[str, object] | None:
    white, black = headers.get("White"), headers.get("Black")
    white_elo, black_elo = parse_int(headers.get("WhiteElo")), parse_int(headers.get("BlackElo"))
    if white in (None, "?") or black in (None, "?") or white_elo is None or black_elo is None:
        return None

    san, clocks, evals, mates = parse_movetext(movetext)
    initial, increment = parse_time_control(headers.get("TimeControl"))
    result = headers.get("Result")
    event = headers.get("Event")
    variant = headers.get("Variant")

    row: dict[str, object] = {
        "game_id": game_index,
        "lichess_id": extract_lichess_id(headers.get("Site")),
        "event": event,
        "site": headers.get("Site"),
        "date": headers.get("UTCDate") or headers.get("Date"),
        "time": headers.get("UTCTime"),
        "round": headers.get("Round"),
        "white": white,
        "black": black,
        "result": result,
        "result_white_score": parse_result_score(result),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "avg_elo": (white_elo + black_elo) / 2,
        "elo_diff": white_elo - black_elo,
        "abs_elo_diff": abs(white_elo - black_elo),
        "white_rating_diff": parse_int(headers.get("WhiteRatingDiff")),
        "black_rating_diff": parse_int(headers.get("BlackRatingDiff")),
        "white_title": headers.get("WhiteTitle"),
        "black_title": headers.get("BlackTitle"),
        "eco": headers.get("ECO"),
        "eco_family": (headers.get("ECO") or "")[:1] or None,
        "opening": headers.get("Opening"),
        "time_control": headers.get("TimeControl"),
        "initial_seconds": initial,
        "increment_seconds": increment,
        "speed": classify_time_control(initial, increment),
        "termination": headers.get("Termination"),
        "variant": variant,
        "rated": str(event or "").lower().startswith("rated"),
        "standard": variant in (None, "", "Standard"),
        "num_halfmoves": len(san),
        "num_fullmoves": (len(san) + 1) // 2,
        "num_white_moves": (len(san) + 1) // 2,
        "num_black_moves": len(san) // 2,
        "num_clock_annotations": sum(v is not None for v in clocks),
        "clock_coverage": sum(v is not None for v in clocks) / len(san) if san else 0.0,
        "num_eval_annotations": sum((e is not None) or (m is not None) for e, m in zip(evals, mates)),
        "eval_coverage": sum((e is not None) or (m is not None) for e, m in zip(evals, mates)) / len(san) if san else 0.0,
    }

    if include_moves_san:
        row["moves_san"] = " ".join(san)
    if include_annotation_series:
        row["clock_seconds_by_ply"] = serialize_list(clocks)
        row["eval_pawns_by_ply"] = serialize_list(evals)
        row["eval_mate_by_ply"] = serialize_list(mates)
    if include_moves_pgn:
        row["moves_pgn"] = movetext

    return row


def iter_rows(
    pgn_path: str | Path,
    *,
    include_moves_san: bool = True,
    include_annotation_series: bool = True,
    include_moves_pgn: bool = False,
    max_games: int | None = None,
) -> Iterator[dict[str, object]]:
    n = 0
    for game_idx, (headers, movetext) in enumerate(iter_pgn_games(pgn_path)):
        if max_games is not None and n >= max_games:
            break
        row = game_to_row(
            headers,
            movetext,
            game_index=game_idx,
            include_moves_san=include_moves_san,
            include_annotation_series=include_annotation_series,
            include_moves_pgn=include_moves_pgn,
        )
        if row is not None:
            n += 1
            yield row


def convert_pgn_to_csv(
    pgn_path: str | Path,
    csv_path: str | Path,
    *,
    chunk_size: int = 100_000,
    include_moves_san: bool = True,
    include_annotation_series: bool = True,
    include_moves_pgn: bool = False,
    max_games: int | None = None,
    show_progress: bool = True,
) -> dict[str, int | str]:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[dict[str, object]] = []
    wrote_header = False
    n = 0

    rows = iter_rows(
        pgn_path,
        include_moves_san=include_moves_san,
        include_annotation_series=include_annotation_series,
        include_moves_pgn=include_moves_pgn,
        max_games=max_games,
    )
    if show_progress and tqdm is not None:
        rows = tqdm(rows, total=max_games, unit="game", desc="PGN -> CSV")

    for row in rows:
        buffer.append(row)
        n += 1
        if len(buffer) >= chunk_size:
            pd.DataFrame(buffer).to_csv(csv_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)
            wrote_header = True
            if show_progress:
                print(f"wrote {n:,} parsed games to {csv_path}", flush=True)
            buffer.clear()

    if buffer or not wrote_header:
        pd.DataFrame(buffer).to_csv(csv_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)

    return {"rows_written": n, "csv_path": str(csv_path)}


def _bool_mask(series: pd.Series) -> pd.Series:
    """Robust boolean mask for bool or string CSV columns."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _filter_sample_chunk(chunk: pd.DataFrame, *, filter_10_0_rapid: bool) -> pd.DataFrame:
    """Apply the project sampling filter before random sampling."""
    if not filter_10_0_rapid:
        return chunk

    mask = pd.Series(True, index=chunk.index)

    if "initial_seconds" in chunk:
        mask &= pd.to_numeric(chunk["initial_seconds"], errors="coerce").eq(600)
    if "increment_seconds" in chunk:
        mask &= pd.to_numeric(chunk["increment_seconds"], errors="coerce").eq(0)
    if "rated" in chunk:
        mask &= _bool_mask(chunk["rated"])
    if "standard" in chunk:
        mask &= _bool_mask(chunk["standard"])

    return chunk.loc[mask]


def sample_csv(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    sample_size: int = 300_000,
    random_state: int = 42,
    chunksize: int = 100_000,
    show_progress: bool = True,
    filter_10_0_rapid: bool = False,
) -> dict[str, int | str]:
    """
    Uniformly sample rows from an existing full CSV.

    This uses two chunked passes over the CSV rather than slow row-by-row
    reservoir sampling. It is intended for reusing an already-created full CSV,
    e.g. to create a 1M training sample without reparsing the original PGN.

    Set ``filter_10_0_rapid=True`` to sample only rated standard 10+0 games.
    That is usually better than taking a random all-time-control sample and
    filtering later.
    """
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: count eligible rows.
    rows_seen = 0
    eligible_rows = 0
    chunks = pd.read_csv(input_csv_path, chunksize=chunksize)
    if show_progress and tqdm is not None:
        chunks = tqdm(chunks, unit="chunk", desc="Counting eligible CSV rows")
    for chunk in chunks:
        rows_seen += len(chunk)
        eligible_rows += len(_filter_sample_chunk(chunk, filter_10_0_rapid=filter_10_0_rapid))

    n_sample = min(int(sample_size), int(eligible_rows))
    if n_sample == 0:
        pd.DataFrame().to_csv(output_csv_path, index=False)
        return {
            "rows_seen": rows_seen,
            "eligible_rows": eligible_rows,
            "sample_rows_written": 0,
            "sample_csv_path": str(output_csv_path),
        }

    rng = np.random.default_rng(random_state)
    target_positions = np.sort(rng.choice(eligible_rows, size=n_sample, replace=False))

    # Pass 2: collect selected eligible rows by their eligible-row position.
    selected: list[pd.DataFrame] = []
    offset = 0
    pos0 = 0
    chunks = pd.read_csv(input_csv_path, chunksize=chunksize)
    if show_progress and tqdm is not None:
        chunks = tqdm(chunks, unit="chunk", desc="Writing sampled CSV")
    for chunk in chunks:
        filtered = _filter_sample_chunk(chunk, filter_10_0_rapid=filter_10_0_rapid).reset_index(drop=True)
        n = len(filtered)
        if n == 0:
            continue

        lo = offset
        hi = offset + n
        while pos0 < len(target_positions) and target_positions[pos0] < lo:
            pos0 += 1
        pos1 = pos0
        while pos1 < len(target_positions) and target_positions[pos1] < hi:
            pos1 += 1

        if pos1 > pos0:
            local = target_positions[pos0:pos1] - offset
            selected.append(filtered.iloc[local])

        offset += n
        pos0 = pos1

    out = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    out.to_csv(output_csv_path, index=False)

    return {
        "rows_seen": rows_seen,
        "eligible_rows": eligible_rows,
        "sample_rows_written": len(out),
        "sample_csv_path": str(output_csv_path),
        "filter_10_0_rapid": filter_10_0_rapid,
    }


def convert_pgn_to_full_and_sample_csv(
    pgn_path: str | Path,
    full_csv_path: str | Path,
    sample_csv_path: str | Path,
    *,
    sample_size: int = 300_000,
    random_state: int = 42,
    chunk_size: int = 100_000,
    full_include_moves_pgn: bool = False,
    sample_include_moves_pgn: bool = True,
    max_games: int | None = None,
    show_progress: bool = True,
) -> dict[str, int | str]:
    """Single-pass conversion: write full CSV and a uniform random 300K sample."""
    full_csv_path = Path(full_csv_path)
    sample_csv_path = Path(sample_csv_path)
    full_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)
    reservoir: list[dict[str, object]] = []
    buffer: list[dict[str, object]] = []
    wrote_header = False
    n = 0

    rows = iter_rows(
        pgn_path,
        include_moves_san=True,
        include_annotation_series=True,
        include_moves_pgn=full_include_moves_pgn or sample_include_moves_pgn,
        max_games=max_games,
    )
    if show_progress and tqdm is not None:
        rows = tqdm(rows, total=max_games, unit="game", desc="PGN -> full CSV + 300K sample")

    for row in rows:
        n += 1
        full_row = row.copy()
        if not full_include_moves_pgn:
            full_row.pop("moves_pgn", None)
        buffer.append(full_row)

        sample_row = row.copy()
        if not sample_include_moves_pgn:
            sample_row.pop("moves_pgn", None)
        if len(reservoir) < sample_size:
            reservoir.append(sample_row)
        else:
            j = int(rng.integers(0, n))
            if j < sample_size:
                reservoir[j] = sample_row

        if len(buffer) >= chunk_size:
            pd.DataFrame(buffer).to_csv(full_csv_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)
            wrote_header = True
            if show_progress:
                print(f"wrote {n:,} parsed games to {full_csv_path}; reservoir currently has {len(reservoir):,} games", flush=True)
            buffer.clear()

    if buffer or not wrote_header:
        pd.DataFrame(buffer).to_csv(full_csv_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)

    if show_progress:
        print(f"finished PGN scan; writing random sample to {sample_csv_path}", flush=True)
    sample = pd.DataFrame(reservoir)
    if len(sample):
        sample = sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sample.to_csv(sample_csv_path, index=False)

    return {
        "input_path": str(pgn_path),
        "full_csv_path": str(full_csv_path),
        "sample_csv_path": str(sample_csv_path),
        "parsed_games": n,
        "sample_games": len(sample),
        "random_state": random_state,
    }
