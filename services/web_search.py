from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import re
import json
import logging
import asyncio
import os
from autogen import UserProxyAgent, AssistantAgent
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from config.settings import settings

import nest_asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebPage:
    """Individual web page result with metadata"""
    title: str
    url: str
    summary: str
    metadata: Dict[str, str] = field(default_factory=dict)  # Additional metadata like date, author, etc.

@dataclass
class SearchResult:
    """Structured search result containing multiple web pages"""
    query: str
    results: List[WebPage]
    error: Optional[str] = None

    def format_discord(self) -> str:
        """Format the result for Discord output"""
        try:
            if self.error:
                logger.error(f"Formatting error result: {self.error}")
                return f"**Error:** {self.error}"

            sections = []
            logger.debug(f"Formatting search results. Query: {self.query}, Number of results: {len(self.results)}")
            
            # Add search query header
            sections.append(f"**Search Results for:** {self.query}")
            
            # Just add the URLs to let Discord handle the link previews
            for i, result in enumerate(self.results, 1):
                logger.debug(f"Adding URL for result {i}: {result.url}")
                sections.append(result.url)
            
            # Join all sections with double newlines for spacing
            formatted = "\n\n".join(sections)
            logger.debug(f"Final formatted output length: {len(formatted)}")
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format Discord message: {e}", exc_info=True)
            return "Error: Failed to format search results"

class WebSearchService:
    def __init__(self):
        """Initialize the web search service"""
        logger.info("Initializing WebSearchService...")
        try:
            # Configure LLM based on provider
            config_list = []
            
            if settings.LLM_PROVIDER == "lmstudio":
                config_list.append({
                    "base_url": settings.LLM_BASE_URL,
                    "api_key": "not-needed",  # LMStudio doesn't need an API key
                    "model": settings.LLM_MODEL
                })
            elif settings.LLM_PROVIDER == "openai":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key not configured")
                config_list.append({
                    "base_url": settings.OPENAI_API_BASE,
                    "api_key": settings.OPENAI_API_KEY,
                    "model": settings.OPENAI_MODEL
                })
            elif settings.LLM_PROVIDER == "gemini":
                if not settings.GEMINI_API_KEY:
                    raise ValueError("Gemini API key not configured")
                config_list.append({
                    "api_key": settings.GEMINI_API_KEY,
                    "model": settings.GEMINI_MODEL
                })
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
            
            logger.info(f"Using LLM provider: {settings.LLM_PROVIDER}")
            
            llm_config = {
                "config_list": config_list,
                "temperature": 0.7,
                "timeout": settings.API_TIMEOUT or 30,
                "cache_seed": 42,  # Enable response caching
                "max_retries": settings.MAX_RETRIES or 3
            }
            
            # Create agents
            self.user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )

            self.assistant = AssistantAgent(
                name="assistant",
                llm_config=llm_config,
                system_message="""You are a web research assistant.
                Your primary task is to analyze web search results and create detailed summaries.
                
                IMPORTANT RULES:
                1. ONLY use information from the provided search results
                2. NEVER make up or hallucinate information
                3. For each result, provide:
                   - The exact title from the search result
                   - The exact URL from the search result
                   - A brief 1-2 sentence summary based on the content
                4. Format responses with [START] and [END] markers
                5. Use double quotes in JSON
                6. Keep summaries short and focused
                
                Follow these rules strictly and only use real information from search results.
                """
            )
            
            # Configure Chrome options
            options = uc.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            # Initialize undetected-chromedriver
            self.driver = uc.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, 10)  # 10 second wait timeout
            
            logger.info(f"WebSearchService initialized with {settings.LLM_PROVIDER}")
            
        except Exception as e:
            error_msg = f"Failed to initialize WebSearchService: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def search(self, query: str, result_limit: int = 5) -> SearchResult:
        """
        Search for information about the given query using web search.
        
        Args:
            query: The search query
            result_limit: Number of results to return (default: 5, max: 10)
        """
        logger.info(f"Starting web search for query: {query} with limit: {result_limit}")
        try:
            # Ensure result_limit is within bounds
            result_limit = max(1, min(result_limit, 10))
            
            # Navigate to DuckDuckGo
            logger.info("Navigating to DuckDuckGo...")
            self.driver.get("https://html.duckduckgo.com/html/")
            logger.info("Successfully loaded DuckDuckGo")
            
            # Find search box and enter query
            logger.info("Looking for search box...")
            try:
                search_box = self.wait.until(
                    EC.presence_of_element_located((By.NAME, "q"))
                )
                logger.info("Found search box")
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                logger.info("Submitted search query")
            except Exception as e:
                logger.error(f"Failed to interact with search box: {str(e)}\nFull traceback:", exc_info=True)
                raise
            
            # Wait for search results
            logger.info("Waiting for search results...")
            try:
                search_results = self.wait.until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "result"))
                )[:result_limit]  # Get specified number of results
                logger.info(f"Found {len(search_results)} search results")
            except Exception as e:
                logger.error(f"Failed to find search results: {str(e)}\nFull traceback:", exc_info=True)
                raise
            
            # Extract information from results
            results_text = ""
            web_pages = []  # Store WebPage objects
            for i, result in enumerate(search_results):
                try:
                    logger.info(f"Processing result {i+1}...")
                    
                    # Find elements within each result
                    title_elem = result.find_element(By.CLASS_NAME, "result__title")
                    link_elem = result.find_element(By.CLASS_NAME, "result__url")
                    snippet_elem = result.find_element(By.CLASS_NAME, "result__snippet")
                    
                    title = title_elem.text.strip()
                    url = link_elem.get_attribute("href")
                    snippet = snippet_elem.text.strip()
                    
                    # Create WebPage object
                    web_pages.append(WebPage(
                        title=title,
                        url=url,
                        summary=snippet
                    ))
                    
                    logger.info(f"Successfully extracted result {i+1}: {title}")
                    results_text += f"Title: {title}\nURL: {url}\nContent: {snippet}\n\n"
                except Exception as e:
                    logger.error(f"Failed to extract result {i+1}: {str(e)}\nFull traceback:", exc_info=True)
                    continue
            
            if not web_pages:
                logger.error("No valid search results were extracted")
                return SearchResult(query=query, results=[], error="No search results found")
            
            logger.info("Creating analysis prompt...")
            # Create analysis prompt
            search_prompt = (
                f"Here are the search results for: {query}\n\n"
                f"{results_text}\n\n"
                "Analyze these results and format your findings in this exact format:\n\n"
                "[START]\n"
                "{\n"
                f'    "query": "{query}",\n'
                '    "results": [\n'
                '        {\n'
                '            "title": "Result title",\n'
                '            "url": "Result URL",\n'
                '            "summary": "Result summary"\n'
                '        }\n'
                "    ]\n"
                "}\n"
                "[END]\n\n"
                "IMPORTANT:\n"
                "1. Only include information from the search results\n"
                "2. Use real source URLs from the results\n"
                "3. Do not make up information\n"
            )
            
            logger.info("Getting analysis from assistant...")
            # Get analysis from assistant
            chat_result = await asyncio.wait_for(
                self.user_proxy.a_initiate_chat(
                    recipient=self.assistant,
                    message=search_prompt,
                    max_turns=2
                ),
                timeout=30  # 30 second timeout
            )
            
            # Return the original web pages instead of processing chat result
            logger.info(f"Returning {len(web_pages)} search results")
            return SearchResult(
                query=query,
                results=web_pages,
                error=None
            )
            
        except asyncio.TimeoutError:
            logger.error("Search request timed out")
            return SearchResult(query=query, results=[], error="Search request timed out")
        except Exception as e:
            error_msg = f"An error occurred while searching: {str(e)}"
            logger.error(f"{error_msg}\nFull traceback:", exc_info=True)
            return SearchResult(query=query, results=[], error=error_msg)
        finally:
            try:
                # Clean up browser resources
                logger.info("Cleaning up browser resources...")
                self.driver.delete_all_cookies()
                self.driver.execute_script("window.localStorage.clear();")
                self.driver.execute_script("window.sessionStorage.clear();")
                logger.info("Browser cleanup completed")
            except Exception as e:
                logger.warning(f"Failed to clean up browser resources: {str(e)}\nFull traceback:", exc_info=True)

    async def _process_chat_result(self, result: Any) -> Optional[SearchResult]:
        """Process the chat result and extract the JSON response."""
        logger.info("Processing chat result...")
        try:
            # Get the last assistant message from the chat history
            if not result or not hasattr(result, 'chat_history'):
                logger.error("No chat history found in result")
                return None
                
            # Find the last assistant message with content
            assistant_messages = []
            for msg in result.chat_history:
                if isinstance(msg, dict) and msg.get('role') == 'assistant' and msg.get('content'):
                    content = msg['content']
                    # Skip messages that look like thinking process
                    if not content.strip().startswith('<think>'):
                        assistant_messages.append(content)
            
            if not assistant_messages:
                logger.error("No valid assistant messages found")
                return None
                
            # Get the last valid message
            message = assistant_messages[-1]
            logger.debug(f"Processing message content: {message[:200]}...")
            
            # Extract JSON between [START] and [END] markers
            match = re.search(r'\[START\]\s*(\{.*?\})\s*\[END\]', message, re.DOTALL)
            if not match:
                logger.error("No [START]/[END] markers found in message")
                return None
            
            json_str = match.group(1).strip()
            logger.debug(f"Extracted JSON string: {json_str[:200]}...")
            
            parsed = json.loads(json_str)
            
            # Create SearchResult from the parsed JSON
            search_results = []
            for result in parsed.get("results", []):
                search_results.append(
                    WebPage(
                        title=result["title"],
                        url=result["url"],
                        summary=result["summary"]
                    )
                )
            
            logger.info(f"Successfully parsed {len(search_results)} search results")
            return SearchResult(
                query=parsed.get("query", ""),
                results=search_results,
                error=None
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}\nJSON string was: {json_str if 'json_str' in locals() else 'Not available'}")
            return None
        except Exception as e:
            logger.error(f"Error processing chat result: {str(e)}", exc_info=True)
            return None

    def __del__(self):
        """Clean up resources when the service is destroyed"""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
        except Exception as e:
            logger.warning(f"Failed to clean up WebDriver: {e}")

# Create singleton instance
web_search_service = WebSearchService()
