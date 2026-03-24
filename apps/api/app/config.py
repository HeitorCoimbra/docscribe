from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    nextauth_secret: str
    docscribe_db_url: str
    groq_api_key: str
    anthropic_api_key: str
    cors_origins: list[str] = ["http://localhost:3000"]
    max_history_messages: int = 50
    claude_model: str = "claude-sonnet-4-5-20250929"
    whisper_model: str = "whisper-large-v3-turbo"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
