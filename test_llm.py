from llm.providers import get_llm_provider
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for cleaner console output
)
logger = logging.getLogger(__name__)

def test_llm_response():
    # Initialize the LLM provider
    llm_provider = get_llm_provider()
    print(f"\nTesting LLM Provider: {llm_provider.__class__.__name__}\n")
    
    # Test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please introduce yourself and tell me what capabilities you have."}
    ]
    
    try:
        print("Sending request to LLM...")
        response = llm_provider.generate_response(messages, timeout=30)
        print("\nLLM Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == '__main__':
    test_llm_response()
