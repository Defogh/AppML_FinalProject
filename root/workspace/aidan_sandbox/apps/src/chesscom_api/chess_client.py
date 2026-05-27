from typing import Any

import httpx
from fastapi import HTTPException

from chesscom_api.config import settings


class ChessComClient:
  def __init__(self) -> None:
    self.base_url = settings.chesscom_base_url
    self.headers = {
      "User-Agent": settings.user_agent,
      "Accept": "application/json",
      "Accept-Encoding": "gzip",
    }

  async def get_json(self, path: str) -> dict[str, Any]:
    url = f"{self.base_url}{path}"

    async with httpx.AsyncClient(
      timeout=30.0,
      follow_redirects=True,
    ) as client:
      response = await client.get(
        url,
        headers=self.headers,
      )

    if response.status_code == 404:
      raise HTTPException(
        status_code=404,
        detail=f"Not found: {path}",
      )

    if response.status_code == 429:
      raise HTTPException(
        status_code=429,
        detail="Chess.com rate limit reached. Try again later.",
      )

    response.raise_for_status()
    return response.json()

  async def get_text(self, path: str) -> str:
    url = f"{self.base_url}{path}"

    async with httpx.AsyncClient(
      timeout=30.0,
      follow_redirects=True,
    ) as client:
      response = await client.get(
        url,
        headers={
          **self.headers,
          "Accept": "text/plain, application/x-chess-pgn",
        },
      )

    if response.status_code == 404:
      raise HTTPException(
        status_code=404,
        detail=f"Not found: {path}",
      )

    if response.status_code == 429:
      raise HTTPException(
        status_code=429,
        detail="Chess.com rate limit reached. Try again later.",
      )

    response.raise_for_status()
    return response.text

  async def get_player_profile(
    self,
    username: str,
  ) -> dict[str, Any]:
    return await self.get_json(f"/player/{username}")

  async def get_player_archives(
    self,
    username: str,
  ) -> dict[str, Any]:
    return await self.get_json(
      f"/player/{username}/games/archives"
    )

  async def get_player_games_by_month(
    self,
    username: str,
    year: int,
    month: int,
  ) -> dict[str, Any]:
    return await self.get_json(
      f"/player/{username}/games/{year:04d}/{month:02d}"
    )

  async def get_player_games_by_month_pgn(
    self,
    username: str,
    year: int,
    month: int,
  ) -> str:
    return await self.get_text(
      f"/player/{username}/games/{year:04d}/{month:02d}/pgn"
    )