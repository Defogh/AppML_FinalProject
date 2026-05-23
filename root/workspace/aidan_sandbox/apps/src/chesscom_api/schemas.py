# Response models that define what the API returns

from pydantic import BaseModel


class PlayerProfile(BaseModel):
  username: str
  player_id: int | None = None
  title: str | None = None
  status: str | None = None
  name: str | None = None
  followers: int | None = None
  country: str | None = None
  joined: int | None = None
  last_online: int | None = None


class PlayerArchives(BaseModel):
  username: str
  archives: list[str]