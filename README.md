# Autogen Discord Bot

Autogen is an intelligent Discord bot powered by LMStudio and ChromaDB, featuring memory management and natural conversation capabilities. It maintains context through conversations and provides human-like responses while efficiently managing long messages and user interactions.

## Features

- 🤖 Natural conversation using local LLM through LMStudio
- 💭 Long-term memory using ChromaDB with semantic search
- 🎯 Smart context retrieval for relevant responses
- 🔒 Server and channel-specific permissions
- 🚫 Automatic @ mention filtering
- ✨ Interactive response indicators
- 📝 Automatic handling of long messages
- 🧠 Improved memory recall with optimized thresholds

## Prerequisites

- Python 3.10 or higher
- LMStudio running locally (or another OpenAI-compatible API endpoint)
- Discord Bot Token and Application
- CUDA-capable GPU (recommended for LMStudio)

## Installation

### Option 1: Using Conda (Recommended)

1. Create a new conda environment:
```bash
conda create -n autogen python=3.10
conda activate autogen
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/autogen-bot.git
cd autogen-bot
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Using Python venv

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/autogen-bot.git
cd autogen-bot
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root with the following variables:
```env
# Discord Configuration
DISCORD_TOKEN=your_discord_bot_token
ALLOWED_SERVER_ID=your_server_id
ALLOWED_CHANNEL_ID=your_channel_id

# LLM Configuration
LLM_ENDPOINT=http://localhost:1234/v1
LLM_MODEL=your_model_name

# Memory Configuration
CHROMA_PERSIST_DIR=./data/chroma

# API Configuration
API_TIMEOUT=30
MAX_RETRIES=3

# Memory Settings
MAX_MEMORIES_PER_QUERY=15
CONTEXT_WINDOW_SIZE=10
MEMORY_SIMILARITY_THRESHOLD=0.15  # Threshold for memory relevance (0.0 to 1.0)
TOP_MEMORIES_TO_CONSIDER=8        # Number of top memories to consider
```

2. Update the configuration values:
   - Get your Discord bot token from [Discord Developer Portal](https://discord.com/developers/applications)
   - Find your server ID by enabling Developer Mode in Discord and right-clicking your server
   - Find your channel ID by right-clicking the channel you want to use

## Running LMStudio

1. Download and install [LMStudio](https://lmstudio.ai/)
2. Load your preferred model (recommended: deepseek-r1-distill-qwen-14b)
3. Start the local server with the following settings:
   - Host: 0.0.0.0
   - Port: 1234
   - Context Length: 4096 (or model maximum)
   - Temperature: 0.7
   - Max Tokens: 8000 (for handling longer responses)

## Features in Detail

### Interactive Response Indicators
- Shows thinking indicators while processing responses
- Variety of engaging status messages with emojis
- Updates messages in place for a clean chat experience

### Smart Memory Management
- Improved memory recall with optimized similarity thresholds
- Retrieves up to 8 relevant memories per query
- Maintains conversation context effectively

### Message Handling
- Automatically splits long responses into multiple messages
- Preserves message formatting and readability
- Handles messages up to Discord's character limit efficiently

### @ Mention Handling
- Ignores messages containing @ mentions
- Allows for natural conversation flow without bot interference
- Perfect for mixed conversations with other users

## Project Structure

```
autogen-bot/
├── bot/
│   ├── __init__.py
│   └── discord_bot.py      # Discord bot implementation with message handling
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration and environment variables
├── llm/
│   ├── __init__.py
│   └── llm_client.py       # LLM integration with improved response handling
├── memory/
│   ├── __init__.py
│   └── memory_manager.py   # Memory management with optimized recall
├── utils/
│   ├── __init__.py
│   └── embeddings.py       # Embedding utilities for semantic search
├── .env
├── main.py
├── requirements.txt
└── README.md
```

## Usage

The bot provides a natural chat experience:
- Responds to messages in configured channels
- Shows thinking indicators while processing
- Automatically splits long responses
- Ignores @ mentions for better conversation flow
- Maintains context through ChromaDB memory system

Example interaction:
```
User: Tell me about quantum computing
Bot: 🤔 Let me think about that...
[Bot updates with comprehensive response]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LMStudio](https://lmstudio.ai/) for local LLM hosting
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [discord.py](https://discordpy.readthedocs.io/) for Discord integration
