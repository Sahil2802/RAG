from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-docs"
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"

    # Groq
    GROQ_API_KEY: str

    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    # LangSmith (optional)
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "rag-app"

    # App
    FRONTEND_URL: str = "http://localhost:5173"
    MAX_FILE_SIZE_MB: int = 50

    # Retrieval tuning
    RETRIEVAL_TOP_K_PER_RETRIEVER: int = 20
    RETRIEVAL_FINAL_TOP_N: int = 5
    RRF_K_CONSTANT: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
