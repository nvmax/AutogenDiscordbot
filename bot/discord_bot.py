import discord
import re
from typing import Optional, List
from discord.ext import commands
from config.settings import settings
from memory.memory_manager import memory_manager
from llm.llm_client import llm_client
import random

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
    
    async def setup_hook(self):
        """Setup hook that runs when the bot is ready."""
        print(f"Logged in as {self.user}")
        print("Bot is ready!")
    
    def is_allowed_channel(self, channel_id: int) -> bool:
        """Check if the channel is allowed."""
        return channel_id == settings.ALLOWED_CHANNEL_ID
    
    def is_allowed_server(self, guild_id: int) -> bool:
        """Check if the server is allowed."""
        return guild_id == settings.ALLOWED_SERVER_ID
    
    def should_respond(self, content: str) -> bool:
        """Determine if the bot should respond to this message."""
        # Don't respond to messages containing @
        return '@' not in content
    
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
                # Add word to current chunk
                if current_chunk:
                    current_chunk += " "
                current_chunk += word
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

bot = AG2Bot()

@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Check if message is in allowed server and channel
    if not (bot.is_allowed_server(message.guild.id) and 
            bot.is_allowed_channel(message.channel.id)):
        return
    
    # Ignore any message containing @
    if not bot.should_respond(message.content):
        return
    
    try:
        # Send thinking message
        thinking_msg = await message.channel.send(random.choice(bot.thinking_messages))
        
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
        error_message = f"An error occurred: {str(e)}"
        # If thinking message exists, update it with error
        try:
            await thinking_msg.edit(content=error_message)
        except:
            await message.channel.send(error_message)

def run_bot():
    """Run the Discord bot."""
    bot.run(settings.DISCORD_TOKEN)
