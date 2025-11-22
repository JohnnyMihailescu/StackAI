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
    cohere_embed_model: str = "embed-english-v3.0"
    cohere_batch_size: int = 96  # Cohere's limit per API call

    # Storage
    data_dir: str = "data"

    # Batch processing limits
    max_batch_size: int = 500  # Max chunks per batch request

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


settings = Settings()
