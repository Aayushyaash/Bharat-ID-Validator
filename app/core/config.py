from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "Id Validator Service"
    API_V1_STR: str = "/api/v1"
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    BASE_MODEL_PATH: Path = BASE_DIR / "models"
    MODEL_CONFIG_PATH: Path = BASE_MODEL_PATH / "config.json"
    
    # Validation
    CONFIDENCE_THRESHOLD: float = 0.98

    # Logging
    LOG_EXCLUDED_PATHS: set[str] = {"/metrics", "/", "/health", "/ready"}

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")

settings = Settings()