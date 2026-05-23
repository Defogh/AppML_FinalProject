import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from chesscom_api.chess_client import ChessComClient
from chesscom_api.schemas import PlayerArchives, PlayerProfile

app = FastAPI(title="Chess.com Public Data API")

client = ChessComClient()


# Make username normal
def normalize_username(username: str) -> str:
  username = username.strip().lower()

  if not re.fullmatch(r"[a-z0-9_-]{1,40}", username):
    raise HTTPException(
      status_code=400,
      detail="Invalid Chess.com username",
    )

  return username


# Safety check for month numbers
def validate_month(month: int) -> None:
  if month < 1 or month > 12:
    raise HTTPException(
      status_code=400,
      detail="Month must be between 1 and 12.",
    )


# Check that API is alive and well
@app.get("/")
def health_check() -> dict[str, str]:
  return {"status": "ok"}


# API call
@app.get("/players/{username}", response_model=PlayerProfile)
async def get_player(username: str) -> PlayerProfile:
  username = normalize_username(username)
  data = await client.get_player_profile(username)

  return PlayerProfile(
    username=data["username"],
    player_id=data.get("player_id"),
    title=data.get("title"),
    status=data.get("status"),
    name=data.get("name"),
    followers=data.get("followers"),
    country=data.get("country"),
    joined=data.get("joined"),
    last_online=data.get("last_online"),
  )

# Define API checkpoint for a given chess.com username
@app.get(
  "/players/{username}/archives",
  response_model=PlayerArchives,
)
async def get_player_archives(username: str) -> PlayerArchives:
  username = normalize_username(username)
  data = await client.get_player_archives(username)

  return PlayerArchives(
    username=username,
    archives=data.get("archives", []),
  )


# Get player games by month
@app.get("/players/{username}/games/{year}/{month}")
async def get_player_games_by_month(
  username: str,
  year: int,
  month: int,
) -> dict:
  username = normalize_username(username)
  validate_month(month)

  return await client.get_player_games_by_month(
    username,
    year,
    month,
  )


# Get player games by month in pgn format
@app.get(
  "/players/{username}/games/{year}/{month}/pgn",
  response_class=PlainTextResponse,
)
async def get_player_games_by_month_pgn(
  username: str,
  year: int,
  month: int,
) -> str:
  username = normalize_username(username)
  validate_month(month)

  return await client.get_player_games_by_month_pgn(
    username,
    year,
    month,
  )