from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import requests
import google.generativeai as genai
from openai import OpenAI
from config.settings import settings
import logging
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], timeout: int) -> str:
        """Generate a response from the LLM provider."""
        pass

class LMStudioProvider(LLMProvider):
    """LMStudio provider implementation."""
    
    def __init__(self):
        self.endpoint = settings.LLM_BASE_URL
        self.model = settings.LLM_MODEL
    
    def generate_response(self, messages: List[Dict[str, str]], timeout: int) -> str:
        response = requests.post(
            f"{self.endpoint}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 8196
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        self.model = settings.OPENAI_MODEL
    
    def generate_response(self, messages: List[Dict[str, str]], timeout: int) -> str:
        try:
            logger.debug(f"Sending request to OpenAI with messages: {json.dumps(messages, indent=2)}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                timeout=timeout
            )
            content = response.choices[0].message.content
            logger.debug(f"Got response from OpenAI: {content}")
            return content
        except Exception as e:
            logger.error(f"Error communicating with OpenAI: {str(e)}", exc_info=True)
            raise Exception(f"Error communicating with OpenAI: {str(e)}")

class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL,
            generation_config={
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
        )
        logger.info(f"Initialized Gemini provider with model: {settings.GEMINI_MODEL}")
    
    def generate_response(self, messages: List[Dict[str, str]], timeout: int) -> str:
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            system_content = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_content += msg["content"] + "\n"
                else:
                    gemini_messages.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [msg["content"]]
                    })
            
            # If we have system content, prepend it to the first user message
            if system_content and gemini_messages:
                first_msg = gemini_messages[0]
                if first_msg["role"] == "user":
                    first_msg["parts"][0] = f"{system_content}\n\nUser message:\n{first_msg['parts'][0]}"
            
            # Create chat session
            chat = self.model.start_chat(history=gemini_messages[:-1])
            
            # Get response for the last message
            last_msg = gemini_messages[-1]["parts"][0] if gemini_messages else system_content
            response = chat.send_message(
                last_msg,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }
            )
            
            if not response.text:
                raise Exception("Empty response from Gemini API")
                
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                raise Exception(
                    "Gemini API rate limit exceeded. Please wait a few minutes before trying again. "
                    "If this persists, check your API quota in the Google Cloud Console."
                )
            elif "403" in error_msg:
                raise Exception(
                    "Gemini API access denied. Please check if your API key is valid and has the necessary permissions."
                )
            else:
                raise Exception(f"Gemini API error: {error_msg}")

def get_llm_provider() -> LLMProvider:
    """Factory function to get the configured LLM provider."""
    providers = {
        "lmstudio": LMStudioProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider
    }
    
    provider_class = providers.get(settings.LLM_PROVIDER.lower())
    if not provider_class:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
    
    return provider_class()
