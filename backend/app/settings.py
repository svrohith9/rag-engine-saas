from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int

    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    gemini_embed_model: str

    upload_dir: Path
    db_path: Path


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    load_dotenv(base_dir / ".env", override=True)

    upload_dir = Path(os.getenv("UPLOAD_DIR", "./data/uploads")).expanduser()
    db_path = Path(os.getenv("DB_PATH", "./data/rag.sqlite3")).expanduser()

    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
        upload_dir=upload_dir,
        db_path=db_path,
    )
