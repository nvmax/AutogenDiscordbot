from typing import Dict, List
from decouple import config

class Settings:
    # Discord Configuration
    DISCORD_TOKEN: str = config('DISCORD_TOKEN')
    ALLOWED_SERVER_ID: int = config('ALLOWED_SERVER_ID', cast=int)
    ALLOWED_CHANNEL_ID: int = config('ALLOWED_CHANNEL_ID', cast=int)
    
    # LLM Configuration
    LLM_PROVIDER: str = config('LLM_PROVIDER', default='lmstudio')  # Options: lmstudio, openai, gemini
    
    # LMStudio Configuration
    LMSTUDIO_ENDPOINT: str = config('LMSTUDIO_ENDPOINT', default='http://192.168.1.225:1234/v1')
    LMSTUDIO_MODEL: str = config('LMSTUDIO_MODEL', default='deepseek-r1-distill-qwen-14b')
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = config('OPENAI_API_KEY', default='')
    OPENAI_MODEL: str = config('OPENAI_MODEL', default='gpt-3.5-turbo')
    
    # Google Gemini Configuration
    GEMINI_API_KEY: str = config('GEMINI_API_KEY', default='')
    GEMINI_MODEL: str = config('GEMINI_MODEL', default='gemini-pro')

    # Memory Configuration
    CHROMA_PERSIST_DIR: str = config('CHROMA_PERSIST_DIR', default='./data/chroma')
    EMBEDDINGS_MODEL: str = config('EMBEDDINGS_MODEL', default='sentence-transformers/all-mpnet-base-v2')
    MAX_MEMORIES_PER_QUERY: int = config('MAX_MEMORIES_PER_QUERY', default=10, cast=int)  # Number of memories to retrieve per query
    CONTEXT_WINDOW_SIZE: int = config('CONTEXT_WINDOW_SIZE', default=5, cast=int)  # Number of recent messages to include in context
    MEMORY_SIMILARITY_THRESHOLD: float = config('MEMORY_SIMILARITY_THRESHOLD', default=0.15, cast=float)
    TOP_MEMORIES_TO_CONSIDER: int = config('TOP_MEMORIES_TO_CONSIDER', default=8, cast=int)
    
    # API Configuration
    API_TIMEOUT: int = config('API_TIMEOUT', default=30, cast=int)
    MAX_RETRIES: int = config('MAX_RETRIES', default=3, cast=int)

settings = Settings()
