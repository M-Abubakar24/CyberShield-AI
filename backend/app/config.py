
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "CyberShield AI"
    APP_ENV: str = "development"
    DEBUG: bool = True

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    CHROMA_PATH: str = "./data/chroma_db"
    UPLOAD_DIR: str = "./data/uploads"

    DATABASE_URL: str = "sqlite:///./cybershield.db"

    MAX_FILE_SIZE_MB: int = 10
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def resolve_path(path: str) -> Path:
    result = Path(path)

    if not result.is_absolute():
        result = BASE_DIR / result

    return result


CHROMA_DIR = resolve_path(settings.CHROMA_PATH)
UPLOAD_DIR = resolve_path(settings.UPLOAD_DIR)

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
