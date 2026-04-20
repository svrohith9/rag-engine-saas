from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int

    cors_allow_origins: list[str]

    # LLM Provider configuration
    llm_provider: str
    
    # Gemini
    gemini_api_key: str
    gemini_model: str
    gemini_embed_model: str
    
    # OpenAI
    openai_api_key: str
    openai_model: str
    openai_embed_model: str
    
    # Anthropic
    anthropic_api_key: str
    anthropic_model: str
    
    # Ollama (local)
    ollama_base_url: str
    ollama_model: str
    
    # Advanced features
    enable_streaming: bool
    enable_reranking: bool
    reranker_model: str
    
    # Web search
    enable_web_search: bool
    tavily_api_key: str
    
    # Rate limiting
    rate_limit_requests_per_minute: int
    
    # Storage
    upload_dir: Path
    db_path: Path


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    load_dotenv(base_dir / ".env", override=True)

    upload_dir = Path(os.getenv("UPLOAD_DIR", "./data/uploads")).expanduser()
    db_path = Path(os.getenv("DB_PATH", "./data/rag.sqlite3")).expanduser()
    cors_allow_origins = [
        o.strip()
        for o in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(",")
        if o.strip()
    ]

    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        cors_allow_origins=cors_allow_origins,
        
        # LLM Provider (gemini, openai, anthropic, ollama)
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
        
        # Gemini
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
        
        # OpenAI
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        
        # Anthropic
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        
        # Ollama
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        
        # Advanced features
        enable_streaming=os.getenv("ENABLE_STREAMING", "true").lower() == "true",
        enable_reranking=os.getenv("ENABLE_RERANKING", "false").lower() == "true",
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        
        # Web search
        enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true",
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        
        # Rate limiting
        rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
        
        upload_dir=upload_dir,
        db_path=db_path,
    )
