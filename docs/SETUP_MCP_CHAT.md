# MCP Chat Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
uv sync
# or
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Open the `.env` file and choose your AI provider:

#### Option A: OpenAI (GPT models)

```bash
# .env file
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=  # Leave commented out for OpenAI
```

**Get your API key:** https://platform.openai.com/api-keys

#### Option B: Google Gemini (Recommended - Free tier available!)

```bash
# .env file
OPENAI_API_KEY=your-google-api-key-here
OPENAI_MODEL=gemini-2.0-flash-exp
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

**Get your API key:** https://aistudio.google.com/apikey

**Available Gemini models:**
- `gemini-2.0-flash-exp` - Fast, experimental
- `gemini-1.5-flash` - Production-ready, fast
- `gemini-1.5-pro` - More capable

#### Option C: Local Models (Ollama, LM Studio)

```bash
# .env file
OPENAI_API_KEY=not-needed
OPENAI_MODEL=llama3.2
OPENAI_BASE_URL=http://localhost:11434/v1
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will:
1. Launch the MCP server (`server_demo.py`)
2. Discover available tools
3. Connect to OpenAI with function calling
4. Start the chat interface at http://localhost:8501

## Available MCP Tools

The `server_demo.py` provides these tools:

- **`math.add(a, b)`** - Add two numbers
- **`json.read(path)`** - Read a JSON file
- **`transform.filter_keys(data, keys)`** - Filter dictionary keys

## Example Prompts

Try these in the chat:

```
"What's 42 + 58?"
"Add 100 and 200"
"Read the package.json file"
"Filter only name and version from the data"
```

## Configuration Options

### Switch Between AI Providers

The app supports any OpenAI-compatible API. Just edit `.env`:

**OpenAI:**
```bash
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=
```

**Google Gemini:**
```bash
OPENAI_API_KEY=your-google-api-key
OPENAI_MODEL=gemini-2.0-flash-exp
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

**Local Ollama:**
```bash
OPENAI_API_KEY=not-needed
OPENAI_MODEL=llama3.2
OPENAI_BASE_URL=http://localhost:11434/v1
```

### Use a Different MCP Server

Edit `.env`:
```bash
MCP_SERVER_SCRIPT=proj0/server.py
```

## Troubleshooting

### "No API key" error
- Make sure `.env` file exists in the project root
- Verify `OPENAI_API_KEY` is set correctly
- Restart the Streamlit app

### "MCP server failed to start"
- Check that `server_demo.py` exists
- Verify FastMCP is installed: `pip install fastmcp`
- Check for Python errors in the terminal

### "Tool not found"
- The MCP server lists available tools in the sidebar
- Make sure the server is running correctly
- Check server logs in the terminal

## File Structure

```
pySMC/
├── app.py                 # Main Streamlit chat interface
├── server_demo.py         # MCP server with tools
├── .env                   # Your environment variables (gitignored)
├── .env.example          # Template for .env
└── SETUP_MCP_CHAT.md    # This guide
```

## Security Notes

- ⚠️ **Never commit `.env` to git** - it's in `.gitignore`
- ✅ Share `.env.example` as a template
- ✅ Keep your API keys private
- ✅ Use environment variables for all secrets

