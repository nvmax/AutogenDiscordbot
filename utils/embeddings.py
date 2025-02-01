from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import settings

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDINGS_MODEL)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def get_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.get_embeddings([text])[0]

embedding_manager = EmbeddingManager()
