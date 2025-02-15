from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Discord Configuration
    DISCORD_TOKEN: str
    ALLOWED_SERVER_ID: str
    ALLOWED_CHANNEL_ID: str
    
    # LLM Configuration
    LLM_PROVIDER: str
    LLM_BASE_URL: str
    LLM_MODEL: str
    
    # OpenAI Configuration
    OPENAI_API_BASE: str
    OPENAI_API_KEY: Optional[str]
    OPENAI_MODEL: str
    
    # Google Gemini Configuration
    GEMINI_API_KEY: Optional[str]
    GEMINI_MODEL: str
    
    # Database Configuration
    CHROMA_DB_PATH: str = "./data/chromadb"  # Keep this default for compatibility
    CHROMA_PERSIST_DIR: str = "./data/chromadb"  # Keep this default for compatibility
    
    # Embeddings Configuration
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Memory Settings
    MAX_MEMORIES_PER_QUERY: int = 15
    CONTEXT_WINDOW_SIZE: int = 10
    MEMORY_SIMILARITY_THRESHOLD: float = 0.15
    TOP_MEMORIES_TO_CONSIDER: int = 8
    
    # API Configuration
    API_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure CHROMA_PERSIST_DIR matches CHROMA_DB_PATH
        self.CHROMA_PERSIST_DIR = self.CHROMA_DB_PATH
        # Create the directory if it doesn't exist
        os.makedirs(self.CHROMA_DB_PATH, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
