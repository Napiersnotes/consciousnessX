"""
Production Settings Management
"""
import os
from typing import Dict, List, Optional, Any
from pydantic import BaseSettings, Field, PostgresDsn, validator
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "ConsciousnessX Production"
    ENVIRONMENT: Environment = Environment.PRODUCTION
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://yourdomain.com"]
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_KEY_HEADER: str = "X-API-Key"
    
    # Database
    DATABASE_URL: PostgresDsn = Field(..., env="DATABASE_URL")
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # LLM Configuration
    DEFAULT_LLM_PROVIDER: LLMProvider = LLMProvider.OPENAI
    LLM_CONFIG: Dict[str, Any] = {}
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_BASE_URL: Optional[str] = None
    
    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = "claude-3-opus-20240229"
    
    # Google
    GOOGLE_API_KEY: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    GOOGLE_MODEL: str = "gemini-pro"
    
    # Azure
    AZURE_OPENAI_API_KEY: Optional[str] = Field(None, env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4"
    
    # Local (Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    
    # Vector Database
    QDRANT_URL: str = Field("http://localhost:6333", env="QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = None
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Cache
    CACHE_TTL: int = 3600  # 1 hour
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @validator("LLM_CONFIG", pre=True, always=True)
    def assemble_llm_config(cls, v, values):
        config = {
            LLMProvider.OPENAI: {
                "api_key": values.get("OPENAI_API_KEY"),
                "model": values.get("OPENAI_MODEL"),
                "base_url": values.get("OPENAI_BASE_URL"),
            },
            LLMProvider.ANTHROPIC: {
                "api_key": values.get("ANTHROPIC_API_KEY"),
                "model": values.get("ANTHROPIC_MODEL"),
            },
            LLMProvider.GOOGLE: {
                "api_key": values.get("GOOGLE_API_KEY"),
                "model": values.get("GOOGLE_MODEL"),
            },
            LLMProvider.AZURE: {
                "api_key": values.get("AZURE_OPENAI_API_KEY"),
                "endpoint": values.get("AZURE_OPENAI_ENDPOINT"),
                "deployment": values.get("AZURE_OPENAI_DEPLOYMENT"),
            },
            LLMProvider.LOCAL: {
                "base_url": values.get("OLLAMA_BASE_URL"),
                "model": values.get("OLLAMA_MODEL"),
            },
        }
        return config


settings = Settings()
