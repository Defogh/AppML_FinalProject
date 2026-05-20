import re

from fastapi import FastAPI, HTTPException

from chesscom_api.chess_client import ChessComClient
from chesscom_api.schemas import PlayerArchives, PlayerProfile

app = FastAPI(title="Chess.com Public Data API")

client = ChessComClient()


# String parsing to make username normal
def normalize_username(username: str) -> str:
  username = username.strip().lower()

  if not re.fullmatch(r"[a-z0-9_-]{1,40}", username):
    raise HTTPException(
      status_code=400,
      detail="Invalid Chess.com username",
    )

  return username


# Add a basic endpoint that confirms the API is running
@app.get("/")
def health_check() -> dict[str, str]:
  return {"status": "ok"}


# Get player data
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


# Get player archives data
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