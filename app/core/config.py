from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # LLM — GitHub Models
    GITHUB_TOKEN: str
    LLM_MODEL: str = "openai/gpt-4.1"
    GITHUB_MODELS_ENDPOINT: str = "https://models.github.ai/inference"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Pinecone (replaces ChromaDB)
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "roadmaps"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # Cloudinary
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str

    # RAG
    GENERATED_CACHE_DIR: str = "/data/generated"   # Fly.io volume path
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 8
    SIMILARITY_THRESHOLD: float = 0.4

    # Security
    ADMIN_API_KEY: str   # Secret key to protect admin endpoints

    # API
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    @field_validator('DEBUG', mode='before')
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return False

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from env file


settings = Settings()
