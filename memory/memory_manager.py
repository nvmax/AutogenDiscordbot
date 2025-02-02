from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import EmbeddingFunction
from config.settings import settings
from utils.embeddings import embedding_manager
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformerEmbedding(EmbeddingFunction):
    def __call__(self, input: List[str]) -> List[List[float]]:
        return embedding_manager.get_embeddings(input)

class MemoryManager:
    def __init__(self):
        """Initialize the memory manager with ChromaDB"""
        self.client = None
        self.collection = None
        
        try:
            # Ensure the ChromaDB directory exists
            os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            logger.info(f"Initializing ChromaDB at: {settings.CHROMA_DB_PATH}")
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(
                    name="user_interactions",
                    embedding_function=SentenceTransformerEmbedding()
                )
                logger.info("Found existing collection")
            except Exception as e:
                logger.info("No existing collection found, creating new one")
                # Create new collection
                self.collection = self.client.create_collection(
                    name="user_interactions",
                    embedding_function=SentenceTransformerEmbedding(),
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity for embeddings
                )
            
            logger.info("Memory manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise
    
    def add_memory(self, user_id: str, message: str, role: str) -> None:
        """Add a new memory entry."""
        try:
            # Clean up the message by removing any think tags or response prefixes
            message = self._clean_message(message)
            
            # Add the memory to ChromaDB
            self.collection.add(
                documents=[message],
                metadatas=[{
                    "user_id": user_id,
                    "role": role
                }],
                ids=[f"{user_id}_{role}_{len(self.collection.get()['ids'])}"]
            )
            
            logger.debug(f"Added memory for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
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

            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                where={"user_id": user_id},
                n_results=limit
            )
            
            # Format the results
            memories = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    memories.append({
                        "text": doc,
                        "role": metadata["role"]
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def get_memory_count(self, user_id: Optional[str] = None) -> int:
        """Get the total number of memories stored for a user or all users."""
        try:
            if user_id:
                results = self.collection.get(where={"user_id": user_id})
            else:
                results = self.collection.get()
            return len(results['ids'])
        except Exception as e:
            logger.error(f"Error counting memories: {e}")
            return 0
    
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
            logger.info(f"Cleared memories for {'user ' + user_id if user_id else 'all users'}")
            
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            raise

# Create singleton instance
memory_manager = MemoryManager()
