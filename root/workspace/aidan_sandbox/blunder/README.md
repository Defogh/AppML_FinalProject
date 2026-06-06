# Generated chess feature extraction pipeline

This generated folder is intended to be copied into your project root.

## Dependencies

Add these to your `uv` environment:

```bash
uv add pandas numpy zstandard python-chess pyarrow matplotlib ipykernel
```

`pyarrow` is only required for Parquet export. Use `--format csv` for a
fallback format, though Parquet is strongly preferred.

## Files

```text
src/chess_eval_features/
  parser.py      # streams .pgn.zst files and yields eval games
  extract.py     # creates raw game and per-ply tables
  features.py    # derives eval/material/phase/game features
  plotting.py    # visual diagnostics
  export.py      # batch export logic
  cli.py         # CLI implementation
process_data.py  # project-root CLI entry point
scratch/process_data.ipynb
```

## Example processing command

```bash
uv run python process_data.py \
  --input data/raw/lichess_db_standard_rated_2017-05.pgn.zst \
  --output-dir data/processed/lichess_2017-05_eval_n10000 \
  --n-eval-games 10000 \
  --batch-size 1000 \
  --format parquet
```

The command writes:

```text
games.parquet
plies.parquet
features.parquet
feature_dictionary.csv
export_manifest.json
```
