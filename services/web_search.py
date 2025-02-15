from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import re
import json
import logging
import asyncio
import os
from datetime import datetime
import discord
import aiohttp
from bs4 import BeautifulSoup
import bs4
from urllib.parse import urljoin, urlparse
import requests
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebPage:
    """Individual web page result with metadata"""
    title: str
    url: str
    summary: str
    image_url: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)  # Additional metadata like date, author, etc.

@dataclass
class SearchResult:
    """Structured search result containing multiple web pages"""
    query: str
    results: List[WebPage]
    error: Optional[str] = None

    def format_discord(self) -> List[discord.Embed]:
        """Format search results as Discord embeds"""
        if not self.results:
            return []
            
        embeds = []
        for i, result in enumerate(self.results, 1):
            try:
                # Clean the URL by removing tracking parameters
                cleaned_url = result.url
                if "?" in cleaned_url:
                    cleaned_url = cleaned_url.split("?")[0]
                
                # Create an embed for each result
                embed = discord.Embed(
                    title=result.title[:256],  # Discord has a 256 character limit for titles
                    url=cleaned_url,
                    description=result.summary[:4096] if result.summary else "No summary available",
                    color=0x00b0f4  # Nice blue color
                )
                
                # Add thumbnail image if available
                if result.image_url:
                    embed.set_thumbnail(url=result.image_url)
                    # Also try setting the image in case thumbnail doesn't work
                    embed.set_image(url=result.image_url)
                
                embed.set_footer(text=f"Result {i} of {len(self.results)}")
                
                # Add any additional metadata if available
                if result.metadata:
                    metadata_text = "\n".join([f"**{k}:** {v}" for k, v in result.metadata.items()])
                    if metadata_text:
                        embed.add_field(
                            name="Additional Information",
                            value=metadata_text[:1024],  # Discord has a 1024 character limit for field values
                            inline=False
                        )
                
                embeds.append(embed)
            except Exception as e:
                logger.error(f"Error formatting result {i} as Discord embed: {e}")
                continue
                
        return embeds

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
            
            # Initialize undetected-chromedriver with specific version
            self.driver = uc.Chrome(
                options=options,
                version_main=132  # Match installed Chrome version
            )
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
                results = self.driver.find_elements(By.CLASS_NAME, "result")
                logger.info(f"Found {len(results)} search results")
                
                processed_results = []
                for i, result in enumerate(results[:result_limit], 1):
                    try:
                        logger.info(f"Processing result {i}...")
                        
                        # Extract title and URL
                        title_elem = result.find_element(By.CLASS_NAME, "result__title")
                        title = title_elem.text.strip()
                        url = title_elem.find_element(By.TAG_NAME, "a").get_attribute("href")
                        
                        # Extract summary
                        summary = ""
                        try:
                            summary = result.find_element(By.CLASS_NAME, "result__snippet").text.strip()
                        except:
                            pass
                            
                        # Try to find image
                        image_url = None
                        try:
                            # First try to get preview image from the actual webpage
                            image_url = await self.get_preview_image(url)
                            
                            if not image_url:
                                # Try to find image in search result
                                images = result.find_elements(By.TAG_NAME, "img")
                                for img in images:
                                    src = img.get_attribute("src")
                                    if src and not any(x in src.lower() for x in ['icon', 'logo', 'avatar']):
                                        image_url = src
                                        break
                            
                            # Fallback to favicon if still no image
                            if not image_url:
                                domain = urlparse(url).netloc
                                image_url = f"https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=http://{domain}&size=256"
                                
                        except Exception as e:
                            logger.error(f"Error getting image for result: {e}")
                            # Use favicon as fallback
                            domain = urlparse(url).netloc
                            image_url = f"https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=http://{domain}&size=256"
                        
                        logger.info(f"Successfully extracted result {i}: {title}")
                        processed_results.append(WebPage(
                            title=title,
                            url=url,
                            summary=summary,
                            image_url=image_url
                        ))
                        
                    except Exception as e:
                        logger.error(f"Error processing result {i}: {e}")
                        continue
                
                if not processed_results:
                    logger.error("No valid search results were extracted")
                    return SearchResult(query=query, results=[], error="No search results found")
                
                logger.info(f"Returning {len(processed_results)} search results")
                return SearchResult(
                    query=query,
                    results=processed_results,
                    error=None
                )
                
            except Exception as e:
                logger.error(f"Failed to find search results: {str(e)}\nFull traceback:", exc_info=True)
                raise
            
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

    async def get_preview_image(self, url: str) -> Optional[str]:
        """Get preview image from a URL using Open Graph tags"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Try Open Graph image
                        og_image = soup.find('meta', property='og:image')
                        if og_image:
                            image_url = og_image.get('content')
                            if image_url:
                                # Make sure the URL is absolute
                                return urljoin(url, image_url)
                        
                        # Try Twitter card image
                        twitter_image = soup.find('meta', {'name': 'twitter:image'})
                        if twitter_image:
                            image_url = twitter_image.get('content')
                            if image_url:
                                return urljoin(url, image_url)
                        
                        # Try finding the first large image
                        images = soup.find_all('img')
                        for img in images:
                            src = img.get('src')
                            if src and not any(x in src.lower() for x in ['icon', 'logo', 'avatar']):
                                width = img.get('width')
                                height = img.get('height')
                                if width and height and int(width) >= 200 and int(height) >= 200:
                                    return urljoin(url, src)
                            
        except Exception as e:
            logger.error(f"Error getting preview image for {url}: {e}")
        return None

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
                        summary=result["summary"],
                        image_url=result.get("image_url")
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
