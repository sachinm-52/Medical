from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedVision AI"
    API_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT / 'medvision.db'}"
    
    # File Paths
    UPLOAD_DIR: Path = PROJECT_ROOT / "backend" / "uploads"
    STORAGE_DIR: Path = PROJECT_ROOT / "backend" / "storage"
    CHECKPOINT_DIR: Path = PROJECT_ROOT / "ml" / "checkpoints"
    SAMPLE_DIR: Path = PROJECT_ROOT / "ml" / "samples"
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
settings.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
