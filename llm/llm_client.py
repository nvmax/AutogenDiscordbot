from typing import Dict, List, Optional
import time
import numpy as np
from config.settings import settings
from utils.embeddings import embedding_manager
from .providers import get_llm_provider

class LLMClient:
    def __init__(self):
        self.provider = get_llm_provider()
        self.timeout = settings.API_TIMEOUT
        self.max_retries = settings.MAX_RETRIES
        
        self.system_prompt = (
            "You are Autogen, a friendly and natural conversational AI. Your goal is to chat in a way that feels easy, natural, and helpful. Here's how to keep the conversation flowing:\n\n"
            "1. Keep things sounding natural—don't repeat the same phrases over and over.\n"
            "2. Pay attention to what's already been said so the conversation feels connected and smooth.\n"
            "3. Keep your replies short, clear, and helpful—get to the point without rambling.\n"
            "4. For direct questions, give only the answer without adding extra commentary unless the user asks for more details.\n"
            "5. Don't overcomplicate things. Be concise and direct while staying friendly.\n"
            "6. Stay focused on the current topic. Don't drift off-topic unless the user asks.\n"
            "7. If someone mentions another person, don't assume they're part of the conversation unless it's clear they are.\n"
            "8. Just use plain text—no fancy formatting, markdown, or extra symbols.\n\n"
            "Above all, keep it friendly and conversational, like you're chatting with a friend!"
        )

    def _make_request(self, messages: List[Dict[str, str]], retries: int = 0) -> Optional[str]:
        """Make a request to the LLM provider with retry logic."""
        try:
            content = self.provider.generate_response(messages, self.timeout)
            
            # Remove any thinking process tags and their content
            if "<think>" in content and "</think>" in content:
                start = content.find("<think>")
                end = content.find("</think>") + len("</think>")
                content = content[end:].strip()
            
            return content
            
        except Exception as e:
            if retries < self.max_retries:
                time.sleep(2 ** retries)  # Exponential backoff
                return self._make_request(messages, retries + 1)
            raise Exception(f"Error communicating with LLM: {str(e)}")
    
    def _filter_relevant_memories(self, memories: List[Dict[str, str]], current_topic: str) -> List[Dict[str, str]]:
        """Filter memories to only include those relevant to the current topic."""
        if not memories:
            return []
            
        # Convert memories to a format suitable for embedding comparison
        memory_texts = [mem["text"] for mem in memories]
        
        # Get embeddings for the current topic and memories
        topic_embedding = embedding_manager.get_embeddings([current_topic])[0]
        memory_embeddings = embedding_manager.get_embeddings(memory_texts)
        
        # Calculate similarity scores and pair with indices
        memory_scores = []
        for i, mem_emb in enumerate(memory_embeddings):
            sim = float(np.dot(topic_embedding, mem_emb))
            memory_scores.append((i, sim))
            print(f"Memory {i}: '{memory_texts[i]}' - Similarity: {sim:.3f}")
        
        # Sort memories by similarity score
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N most similar memories above threshold
        threshold = settings.MEMORY_SIMILARITY_THRESHOLD
        top_n = settings.TOP_MEMORIES_TO_CONSIDER
        
        relevant_indices = []
        for idx, score in memory_scores[:top_n]:
            if score > threshold:
                relevant_indices.append(idx)
        
        # Always include the most recent memory for continuity
        if memories and len(memories) - 1 not in relevant_indices:
            relevant_indices.append(len(memories) - 1)
        
        # Sort indices to maintain chronological order
        relevant_indices.sort()
        
        filtered_memories = [memories[i] for i in relevant_indices]
        print(f"Selected {len(filtered_memories)} relevant memories")
        return filtered_memories
    
    def get_response(
        self, 
        user_message: str, 
        context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Get a response from the LLM."""
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add context if provided
        if context:
            # Filter memories for relevance
            print("\nFiltering memories for relevance to:", user_message)
            filtered_context = self._filter_relevant_memories(context, user_message)
            
            if filtered_context:
                # Limit context to configured window size
                recent_context = filtered_context[-settings.CONTEXT_WINDOW_SIZE:]
                context_str = "\n".join([
                    f"{mem['role']}: {mem['text']}" for mem in recent_context
                ])
                messages.append({
                    "role": "system",
                    "content": f"Relevant conversation context:\n{context_str}"
                })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        response = self._make_request(messages)
        
        return response

llm_client = LLMClient()
