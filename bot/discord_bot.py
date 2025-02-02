import discord
import logging
import re
from typing import List, Optional
from discord.ext import commands
from config.settings import settings
from memory.memory_manager import memory_manager
from llm.llm_client import llm_client
from services.web_search import web_search_service
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AG2Bot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        
        # Remove default help command
        self.remove_command('help')
        
        # Discord message length limit
        self.message_limit = 2000
        
        # Thinking messages for variety
        self.thinking_messages = [
            "🤔 Let me think about that...",
            "🧠 Processing your message...",
            "💭 Gathering my thoughts...",
            "⚡ Computing response...",
            "🔄 Analyzing context..."
        ]
        
        # Web search trigger patterns
        self.search_patterns = [
            r"(?i)search for (.+)",
            r"(?i)find information (?:about|on) (.+)",
            r"(?i)look up (.+)",
            r"(?i)what (?:do you know|can you tell me) about (.+)",
            r"(?i)research (.+)"
        ]
        
        logger.info(f"Bot initialized with settings:")
        logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
        logger.info(f"LLM Base URL: {settings.LLM_BASE_URL}")
        logger.info(f"LLM Model: {settings.LLM_MODEL}")
        logger.info(f"Allowed Server: {settings.ALLOWED_SERVER_ID}")
        logger.info(f"Allowed Channel: {settings.ALLOWED_CHANNEL_ID}")
    
    async def setup_hook(self):
        """Setup hook that runs when the bot is ready."""
        logger.info(f"Logged in as {self.user}")
        logger.info("Bot is ready!")
    
    def is_allowed_channel(self, channel_id: int) -> bool:
        """Check if the channel is allowed."""
        allowed_channels = [int(id.strip()) for id in str(settings.ALLOWED_CHANNEL_ID).split(",")]
        is_allowed = channel_id in allowed_channels
        logger.debug(f"Channel {channel_id} allowed: {is_allowed}")
        return is_allowed
    
    def is_allowed_server(self, guild_id: int) -> bool:
        """Check if the server is allowed."""
        allowed_servers = [int(id.strip()) for id in str(settings.ALLOWED_SERVER_ID).split(",")]
        is_allowed = guild_id in allowed_servers
        logger.debug(f"Server {guild_id} allowed: {is_allowed}")
        return is_allowed
    
    def should_respond(self, content: str) -> bool:
        """Determine if the bot should respond to this message."""
        if "@" in content:
            logger.debug("Message contains @, ignoring")
            return False
        return True
    
    def extract_search_query(self, content: str) -> Optional[str]:
        """Extract search query from message if it matches search patterns."""
        for pattern in self.search_patterns:
            match = re.search(pattern, content)
            if match:
                query = match.group(1)
                logger.debug(f"Extracted search query: {query}")
                return query
        return None
    
    def split_message(self, content: str) -> List[str]:
        """Split a long message into chunks that fit within Discord's limit."""
        if len(content) <= self.message_limit:
            return [content]
        
        chunks = []
        current_chunk = ""
        words = content.split()
        
        for word in words:
            # Check if adding the next word would exceed the limit
            if len(current_chunk) + len(word) + 1 > self.message_limit:
                # Add current chunk to chunks list
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " "
                current_chunk += word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

bot = AG2Bot()

@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        logger.debug("Ignoring message from self")
        return
    
    # Check if message is in allowed server and channel
    if not (bot.is_allowed_server(message.guild.id) and 
            bot.is_allowed_channel(message.channel.id)):
        logger.debug("Message not in allowed server/channel")
        return
    
    # Ignore any message containing @
    if not bot.should_respond(message.content):
        logger.debug("Should not respond to message")
        return
    
    try:
        logger.info(f"Processing message: {message.content}")
        
        # Send thinking message
        thinking_msg = await message.channel.send(random.choice(bot.thinking_messages))
        
        # Check if this is a search query
        content = message.content.lower()
        if content.startswith('search for ') or content.startswith('search:'):
            # Handle search query
            search_query = None
            result_limit = 5  # Default limit
            
            if content.startswith('search for '):
                query_text = content.replace('search for ', '', 1).strip()
            else:
                query_text = content.replace('search:', '', 1).strip()
                
            parts = [p.strip() for p in query_text.split(':')]
            search_query = parts[0]
            
            # Check if result limit is specified
            if len(parts) >= 2:
                try:
                    limit = int(parts[1])
                    result_limit = max(1, min(limit, 10))  # Limit between 1 and 10
                except ValueError:
                    pass
                    
            # Perform web search
            await thinking_msg.edit(content="🔍 Searching the web for information...")
            result = await web_search_service.search(search_query, result_limit=result_limit)
            
            if result.error:
                await thinking_msg.edit(content=f"❌ Error: {result.error}")
                return
                
            if not result.results:
                await thinking_msg.edit(content="❌ No results found.")
                return
                
            response = result.format_discord()
            
            # Split and send response
            chunks = bot.split_message(response)
            await thinking_msg.edit(content=chunks[0])
            for chunk in chunks[1:]:
                await message.channel.send(chunk)
                
        else:
            # Handle normal conversation
            # Get relevant memories
            memories = memory_manager.get_relevant_memories(
                str(message.author.id),
                message.content
            )
            
            # Get response from LLM
            response = llm_client.get_response(message.content, memories)
            
            # Store the interaction in memory
            memory_manager.add_memory(
                str(message.author.id),
                message.content,
                "user"
            )
            memory_manager.add_memory(
                str(message.author.id),
                response,
                "assistant"
            )
            
            # Split response into chunks if necessary
            chunks = bot.split_message(response)
            
            # Update thinking message with first chunk
            await thinking_msg.edit(content=chunks[0])
            
            # Send remaining chunks as new messages
            for chunk in chunks[1:]:
                await message.channel.send(chunk)
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        error_message = f"An error occurred: {str(e)}"
        # If thinking message exists, update it with error
        try:
            await thinking_msg.edit(content=error_message)
        except:
            await message.channel.send(error_message)

def run_bot():
    """Run the Discord bot."""
    bot.run(settings.DISCORD_TOKEN)
