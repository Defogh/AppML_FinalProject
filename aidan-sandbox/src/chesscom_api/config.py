from pydantic_settings import BaseSettings


class Settings(BaseSettings):
  chesscom_base_url: str = "https://api.chess.com/pub"
  user_agent: str = (
    "chesscom-learning-api/0.1 "
    "(contact: aidanjsudler@gmail.com)"
  )


settings = Settings()