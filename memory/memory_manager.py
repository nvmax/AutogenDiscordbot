from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import EmbeddingFunction
from config.settings import settings
from utils.embeddings import embedding_manager

class SentenceTransformerEmbedding(EmbeddingFunction):
    def __call__(self, input: List[str]) -> List[List[float]]:
        return embedding_manager.get_embeddings(input)

class MemoryManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create or get collection for user interactions
        self.collection = self.client.get_or_create_collection(
            name="user_interactions",
            embedding_function=SentenceTransformerEmbedding()
        )
    
    def add_memory(self, user_id: str, message: str, role: str) -> None:
        """Add a new memory entry."""
        try:
            # Clean up the message by removing any think tags or response prefixes
            message = self._clean_message(message)
            
            self.collection.add(
                documents=[message],
                metadatas=[{"user_id": user_id, "role": role}],
                ids=[f"{user_id}_{role}_{len(self.collection.get()['ids'])}"]
            )
        except Exception as e:
            raise Exception(f"Error adding memory: {str(e)}")
    
    def _clean_message(self, message: str) -> str:
        """Clean up message by removing think tags and response prefixes."""
        # Remove think tags and their content
        if "<think>" in message and "</think>" in message:
            start = message.find("<think>")
            end = message.find("</think>") + len("</think>")
            message = message[end:].strip()
        
        # Remove "Response:" prefix
        if message.startswith("Response:"):
            message = message[len("Response:"):].strip()
        
        # Remove any leading dashes or newlines
        message = message.lstrip("-\n").strip()
        
        return message
    
    def get_relevant_memories(
        self, 
        user_id: str, 
        query: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Retrieve relevant memories for a user based on query."""
        try:
            # Use configured limit if none provided
            if limit is None:
                limit = settings.MAX_MEMORIES_PER_QUERY

            results = self.collection.query(
                query_texts=[query],
                where={"user_id": user_id},
                n_results=limit
            )
            
            memories = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                memories.append({
                    "text": doc,
                    "role": metadata["role"]
                })
            return memories
        except Exception as e:
            raise Exception(f"Error retrieving memories: {str(e)}")
    
    def get_memory_count(self, user_id: Optional[str] = None) -> int:
        """Get the total number of memories stored for a user or all users."""
        try:
            if user_id:
                results = self.collection.get(where={"user_id": user_id})
            else:
                results = self.collection.get()
            return len(results['ids'])
        except Exception as e:
            raise Exception(f"Error counting memories: {str(e)}")
    
    def clear_memories(self, user_id: Optional[str] = None) -> None:
        """Clear all memories for a specific user or all users."""
        try:
            if user_id:
                # Get all memories for the user
                results = self.collection.get(where={"user_id": user_id})
                if results['ids']:
                    self.collection.delete(ids=results['ids'])
            else:
                # Delete the collection and recreate it
                self.client.delete_collection("user_interactions")
                self.collection = self.client.create_collection(
                    name="user_interactions",
                    embedding_function=SentenceTransformerEmbedding()
                )
        except Exception as e:
            raise Exception(f"Error clearing memories: {str(e)}")

memory_manager = MemoryManager()
