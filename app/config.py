"""Configuration settings for StackAI."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    app_name: str = "StackAI"
    app_version: str = "0.1.0"
    debug: bool = False

    # Cohere API
    cohere_api_key: str

    # Storage
    data_dir: str = "data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


settings = Settings()
