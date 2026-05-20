# Chess.com PGN Export Tool

This project is a small Python/FastAPI tool for downloading public
Chess.com games and exporting them as `.pgn` files.

The current version supports:

- querying Chess.com player profiles
- querying a player's available monthly archives
- querying one player-month of games as JSON
- querying one player-month of games as raw PGN
- exporting many users' games for one month into one `.pgn` file
- skipping duplicate games using the Chess.com `[Link "..."]` PGN tag

The long-term goal is to generate monthly PGN files somewhat similar in
spirit to `database.lichess.org`, using Chess.com's public player archive
data.

## Project structure

```text
.
├── pyproject.toml
├── README.md
├── users.txt
├── data
│   └── chesscom_2024-01.pgn
├── src
│   └── chesscom_api
│       ├── __init__.py
│       ├── chess_client.py
│       ├── config.py
│       ├── export_month.py
│       ├── main.py
│       ├── pgn_tools.py
│       └── schemas.py
└── uv.lock
```

## Setup

This project uses `uv`.

Install dependencies from the lockfile:

```bash
uv sync
```

If setting up from scratch, the main dependencies are:

```bash
uv add fastapi uvicorn httpx pydantic pydantic-settings
```

## Run the API server

From the project root:

```bash
uv run uvicorn chesscom_api.main:app --reload --app-dir src
```

Then open:

```text
http://127.0.0.1:8000
```

You should see:

```json
{
  "status": "ok"
}
```

Interactive FastAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

## API endpoints

### Health check

```http
GET /
```

Example:

```bash
curl http://127.0.0.1:8000/
```

### Get a Chess.com player profile

```http
GET /players/{username}
```

Example:

```bash
curl http://127.0.0.1:8000/players/hikaru
```

### Get a player's monthly archive links

```http
GET /players/{username}/archives
```

Example:

```bash
curl http://127.0.0.1:8000/players/hikaru/archives
```

### Get one player-month as JSON

```http
GET /players/{username}/games/{year}/{month}
```

Example:

```bash
curl http://127.0.0.1:8000/players/hikaru/games/2024/1
```

### Get one player-month as PGN

```http
GET /players/{username}/games/{year}/{month}/pgn
```

Example:

```bash
curl http://127.0.0.1:8000/players/hikaru/games/2024/1/pgn
```

This returns raw PGN text.

## Export many users to one monthly PGN file

The exporter reads a list of Chess.com usernames and downloads each user's
games for a given month.

It writes all unique games to a single `.pgn` file.

### Step 1: Create a username list

Create a file called `users.txt` in the project root.

Example:

```text
hikaru
gothamchess
magnuscarlsen
```

Blank lines are ignored.

Lines starting with `#` are also ignored:

```text
# top players
hikaru
magnuscarlsen

# streamers
gothamchess
```

### Step 2: Run the exporter

Because this project uses a `src/` layout, set `PYTHONPATH=src` when running
the module directly:

```bash
PYTHONPATH=src uv run python -m chesscom_api.export_month \
  --users-file users.txt \
  --year 2024 \
  --month 1 \
  --output data/chesscom_2024-01.pgn
```

This creates:

```text
data/chesscom_2024-01.pgn
```

### Optional: change request delay

The exporter waits between requests to avoid being too aggressive.

Default delay:

```text
1.0 second
```

Use a custom delay:

```bash
PYTHONPATH=src uv run python -m chesscom_api.export_month \
  --users-file users.txt \
  --year 2024 \
  --month 1 \
  --output data/chesscom_2024-01.pgn \
  --delay-seconds 2.0
```

## What the exporter does

For each username, the exporter:

1. Fetches that player's Chess.com PGN archive for the selected month.
2. Splits the multi-game PGN response into individual games.
3. Extracts the `[Link "..."]` tag from each game.
4. Uses that link as the unique game identifier.
5. Skips games already seen.
6. Appends unique games to the output `.pgn` file.

The duplicate check is important because the same game can appear in both
players' monthly archives.

For example, if both `player_a` and `player_b` are in `users.txt`, then their
game against each other may appear twice. The exporter should only write it
once.

## Example output

A generated PGN file will contain games like:

```pgn
[Event "Live Chess"]
[Site "Chess.com"]
[Date "2024.01.31"]
[Round "-"]
[White "Snowflake"]
[Black "Hikaru"]
[Result "0-1"]
[UTCDate "2024.01.31"]
[UTCTime "21:16:57"]
[WhiteElo "2924"]
[BlackElo "3323"]
[TimeControl "180"]
[Termination "Hikaru won by resignation"]
[Link "https://www.chess.com/game/live/100474934539"]

1. e4 g6 2. d4 Bg7 0-1
```

The exact fields are determined by Chess.com's PGN response.

## Common errors

### `ModuleNotFoundError: No module named 'chesscom_api'`

Use `PYTHONPATH=src` when running the exporter:

```bash
PYTHONPATH=src uv run python -m chesscom_api.export_month \
  --users-file users.txt \
  --year 2024 \
  --month 1 \
  --output data/chesscom_2024-01.pgn
```

For the API server, use `--app-dir src`:

```bash
uv run uvicorn chesscom_api.main:app --reload --app-dir src
```

### Output file only has one game

Make sure `pgn_tools.py` deduplicates using the `[Link "..."]` tag, not the
`[Site "..."]` tag.

Chess.com PGNs often use:

```pgn
[Site "Chess.com"]
[Link "https://www.chess.com/game/live/..."]
```

Since `[Site "Chess.com"]` is the same for every game, using `Site` for
deduplication causes almost every game to be skipped.

## Current limitations

This tool does not yet automatically discover large representative username
lists.

Currently, you provide the username list manually with:

```text
users.txt
```

Future extensions may include:

- collecting users from clubs
- collecting titled players
- collecting users from country pages
- collecting opponents from seed users
- rating-stratified username sampling
- writing metadata files for each PGN export
- auditing rating and time-control distributions

## Notes on responsible use

This tool uses Chess.com's public data endpoints. It should not be used to
scrape private data or bypass access controls.

Use reasonable delays between requests and avoid aggressive crawling.

