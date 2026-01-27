from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    
    All settings can be overridden via environment variables or .env file.
    Environment variable names match the setting names (case-sensitive).
    """
    PROJECT_NAME: str = "Id Validator Service"
    API_V1_STR: str = "/api/v1"
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    BASE_MODEL_PATH: Path = BASE_DIR / "models"
    MODEL_CONFIG_PATH: Path = BASE_MODEL_PATH / "config.json"
    
    # Validation
    CONFIDENCE_THRESHOLD: float = 0.98
    
    # File Upload Limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB default
    
    # OCR Processing
    OCR_PADDING_HEIGHT: int = 1   # Vertical padding for OCR fallback
    OCR_PADDING_WIDTH: int = 20   # Horizontal padding for OCR fallback

    # Logging
    LOG_EXCLUDED_PATHS: set[str] = {"/metrics", "/", "/health", "/ready"}

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")


settings = Settings()