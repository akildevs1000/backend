"""
FILE: backend/app/config.py
Application configuration using Pydantic Settings
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Facial Attendance System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=True, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")

    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")

    # Database (SQLite for local dev, PostgreSQL for production)
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./facial_attendance.db",
        env="DATABASE_URL"
    )
    DB_POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=10, env="DB_MAX_OVERFLOW")

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_CACHE_TTL: int = Field(default=3600, env="REDIS_CACHE_TTL")

    # JWT Security
    JWT_SECRET_KEY: str = Field(default="CHANGE_THIS_IN_PRODUCTION_SECRET_KEY_MIN_32_CHARS", env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")

    # InsightFace / AI Model
    INSIGHTFACE_MODEL_NAME: str = Field(default="buffalo_l", env="INSIGHTFACE_MODEL_NAME")
    INSIGHTFACE_CTX_ID: int = Field(default=-1, env="INSIGHTFACE_CTX_ID")  # 0=GPU, -1=CPU
    EMBEDDING_SIZE: int = 512
    FACE_DETECTION_THRESHOLD: float = Field(default=0.5, env="FACE_DETECTION_THRESHOLD")
    FACE_MATCH_THRESHOLD: float = Field(default=0.6, env="FACE_MATCH_THRESHOLD")  # cosine similarity
    MAX_FACES_ALLOWED: int = 1  # Reject images with multiple faces

    # FAISS
    FAISS_INDEX_PATH: str = Field(default="./data/faiss_index.bin", env="FAISS_INDEX_PATH")
    FAISS_INDEX_TYPE: str = Field(default="IVFFlat", env="FAISS_INDEX_TYPE")  # IVFFlat or Flat
    FAISS_NLIST: int = Field(default=100, env="FAISS_NLIST")  # Number of clusters
    FAISS_NPROBE: int = Field(default=10, env="FAISS_NPROBE")  # Search accuracy

    # Image Processing
    MAX_IMAGE_SIZE_MB: int = Field(default=10, env="MAX_IMAGE_SIZE_MB")
    MIN_FACE_SIZE: int = Field(default=80, env="MIN_FACE_SIZE")  # Minimum face pixels
    IMAGE_QUALITY_THRESHOLD: float = Field(default=0.3, env="IMAGE_QUALITY_THRESHOLD")

    # Anti-Spoofing / Liveness
    LIVENESS_ENABLED: bool = Field(default=True, env="LIVENESS_ENABLED")
    LIVENESS_THRESHOLD: float = Field(default=0.7, env="LIVENESS_THRESHOLD")

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=30, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, env="RATE_LIMIT_WINDOW_SECONDS")

    # File Storage
    UPLOAD_DIR: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    PROFILE_PHOTOS_DIR: str = Field(default="./data/photos", env="PROFILE_PHOTOS_DIR")

    # Attendance
    CLOCK_IN_WINDOW_MINUTES: int = Field(default=30, env="CLOCK_IN_WINDOW_MINUTES")
    DUPLICATE_CHECK_MINUTES: int = Field(default=5, env="DUPLICATE_CHECK_MINUTES")

    # CORS
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
