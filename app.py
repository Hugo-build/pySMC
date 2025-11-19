# =============================================
# File: app.py (Streamlit chat host + MCP client)
# =============================================
# A minimal, single-process host that:
# 1) Launches an MCP server via stdio
# 2) Introspects tools via tools/list
# 3) Exposes those tools to an OpenAI Chat Completions model as function-calling tools
# 4) Runs a simple chat UI where the model can call MCP tools one-after-one

import os
import sys
import json
import time
import uuid                          # for unique IDs
import queue                         # for message passing between threads
import threading                     # for concurrent execution
import subprocess                    # for launching MCP servers
from pathlib import Path             # for path operations
from typing import Any, Dict, List   # for type hints

import streamlit as st               # for the simple chat UI
from dotenv import load_dotenv       # for loading .env files

# Load environment variables from .env file
load_dotenv()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # For Gemini or other compatible APIs
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MCP_SERVER_SCRIPT = os.getenv("MCP_SERVER_SCRIPT", "server.py")

# Get workspace root
WORKSPACE_ROOT = Path(__file__).parent.resolve()


# ----------------------------------------------------------------------------
# MCP Client: Handles communication with MCP server via stdio
# ----------------------------------------------------------------------------
class MCPClient:
    """A simple MCP client that communicates with a server via stdio."""
    
    def __init__(self, server_command: List[str], working_directory: str = None):
        """Launch the MCP server as a subprocess."""
        self.process = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=working_directory
        )
        self.request_id = 0
        self.working_directory = working_directory
        
    def _send_request(self, method: str, params: Dict = None) -> Dict:
        """Send a JSON-RPC request and get the response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
        
        # Read response
        response_str = self.process.stdout.readline()
        if not response_str:
            raise Exception("No response from MCP server")
            
        response = json.loads(response_str)
        
        if "error" in response:
            raise Exception(f"MCP Error: {response['error']}")
            
        return response.get("result", {})
    
    def initialize(self):
        """Initialize the MCP connection."""
        return self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "streamlit-chat",
                "version": "1.0.0"
            }
        })
    
    def list_tools(self) -> List[Dict]:
        """Get the list of available tools from the server."""
        result = self._send_request("tools/list")
        return result.get("tools", [])
    
    def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Call a tool on the MCP server."""
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        return result.get("content", [{}])[0].get("text", "")
    
    def close(self):
        """Close the MCP server connection."""
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait(timeout=5)


# ----------------------------------------------------------------------------
# OpenAI Integration
# ----------------------------------------------------------------------------
def convert_mcp_tools_to_openai(mcp_tools: List[Dict]) -> List[Dict]:
    """Convert MCP tool definitions to OpenAI function calling format."""
    openai_tools = []
    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools


def call_openai_chat(messages: List[Dict], tools: List[Dict]) -> Dict:
    """Call OpenAI Chat Completions API (or compatible API like Gemini)."""
    try:
        from openai import OpenAI
        
        # Initialize client with custom base_url if provided (for Gemini, etc.)
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        
        client = OpenAI(**client_kwargs)
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None
        )
        
        return response.choices[0].message
    except ImportError:
        st.error("OpenAI library not installed. Run: pip install openai")
        return None
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None


# ----------------------------------------------------------------------------
# Streamlit Chat UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MCP Chat Demo", page_icon="🤖", layout="wide")
    
    st.title("🤖 MCP Chat Demo")
    st.markdown("Chat with an AI assistant that can use MCP tools!")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Project folder selection (needs to be before MCP client initialization)
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = "proj0"
    
    # Discover available project folders
    available_projects = [d.name for d in WORKSPACE_ROOT.iterdir() 
                         if d.is_dir() and d.name.startswith("proj")]
    
    if not available_projects:
        st.error("⚠️ No project folders found (looking for folders starting with 'proj')")
        st.stop()
    
    # Project selection in sidebar
    with st.sidebar:
        st.header("📁 Project Selection")
        selected_project = st.selectbox(
            "Select a project folder",
            available_projects,
            index=available_projects.index(st.session_state.selected_project) 
                  if st.session_state.selected_project in available_projects else 0,
            key="project_selector"
        )
        
        # If project changed, reset MCP client
        if selected_project != st.session_state.selected_project:
            st.session_state.selected_project = selected_project
            if "mcp_client" in st.session_state:
                try:
                    st.session_state.mcp_client.close()
                except:
                    pass
                del st.session_state.mcp_client
            st.rerun()
        
        project_path = WORKSPACE_ROOT / selected_project
        st.info(f"📂 Working in: `{project_path.name}`")
    
    # Initialize MCP client for selected project
    if "mcp_client" not in st.session_state:
        try:
            project_path = WORKSPACE_ROOT / st.session_state.selected_project
            server_script_path = project_path / MCP_SERVER_SCRIPT
            
            if not server_script_path.exists():
                st.error(f"⚠️ Server script not found: {server_script_path}")
                st.stop()
            
            # Launch MCP server
            with st.spinner(f"Connecting to MCP server in {st.session_state.selected_project}..."):
                st.session_state.mcp_client = MCPClient(
                    [sys.executable, MCP_SERVER_SCRIPT],
                    working_directory=str(project_path)
                )
                st.session_state.mcp_client.initialize()
                
                # Get available tools
                mcp_tools = st.session_state.mcp_client.list_tools()
                st.session_state.mcp_tools = mcp_tools
                st.session_state.openai_tools = convert_mcp_tools_to_openai(mcp_tools)
                
            st.success(f"✅ Connected to {st.session_state.selected_project}! Found {len(mcp_tools)} tools")
        except Exception as e:
            st.error(f"Failed to connect to MCP server: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    # Sidebar: Show available tools
    with st.sidebar:
        st.header("🛠️ Available MCP Tools")
        for tool in st.session_state.mcp_tools:
            with st.expander(f"**{tool['name']}**"):
                st.write(tool.get("description", "No description"))
                st.json(tool.get("inputSchema", {}))
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander("🔧 Tool Calls"):
                    for tc in msg["tool_calls"]:
                        st.code(f"{tc['function']['name']}({tc['function']['arguments']})", language="json")
            # Display any images stored in the message
            # if msg.get("image_base64"):
                # import base64
                # from io import BytesIO
                # from PIL import Image
                
                # image_data = base64.b64decode(msg["image_base64"])
                # image = Image.open(BytesIO(image_data))
                # st.image(image, caption="FE Model Visualization", use_container_width=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Prepare messages for OpenAI
            openai_messages = [
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.messages
            ]
            
            # Call OpenAI
            with st.spinner("Thinking..."):
                response_message = call_openai_chat(
                    openai_messages, 
                    st.session_state.openai_tools
                )
            
            if response_message is None:
                st.error("Failed to get response from OpenAI")
                st.stop()
            
            # Handle tool calls
            if response_message.tool_calls:
                tool_calls = response_message.tool_calls
                tool_results = []
                image_base64_result = None
                
                # Show tool calls
                with st.expander("🔧 Calling tools...", expanded=True):
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        st.write(f"**Calling:** `{tool_name}`")
                        st.json(tool_args)
                        
                        # Execute tool via MCP
                        try:
                            result = st.session_state.mcp_client.call_tool(tool_name, tool_args)
                            tool_results.append(result)
                            
                            # Try to parse result as JSON to check for images
                            try:
                                result_dict = json.loads(result) if isinstance(result, str) else result
                                
                                # Check if result contains a base64 image
                                if isinstance(result_dict, dict) and "image_base64" in result_dict and result_dict["image_base64"]:
                                    st.success(f"**Result:** {result_dict.get('message', 'Success')}")
                                    
                                    # Store image for message history
                                    image_base64_result = result_dict["image_base64"]
                                    
                                    # Display the image
                                    import base64
                                    from io import BytesIO
                                    from PIL import Image
                                    
                                    image_data = base64.b64decode(result_dict["image_base64"])
                                    image = Image.open(BytesIO(image_data))
                                    st.image(image, caption=result_dict.get('message', 'FE Model Plot'), width='content')
                                else:
                                    st.success(f"**Result:** {result}")
                            except (json.JSONDecodeError, TypeError):
                                st.success(f"**Result:** {result}")
                                
                        except Exception as e:
                            st.error(f"Tool execution failed: {str(e)}")
                            result = f"Error: {str(e)}"
                            tool_results.append(result)
                
                # Add tool call results to messages
                assistant_msg = {
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                }
                
                # Add image to message if one was generated
                if image_base64_result:
                    assistant_msg["image_base64"] = image_base64_result
                
                st.session_state.messages.append(assistant_msg)
                
                # Get final response with tool results
                openai_messages.append(assistant_msg)
                for i, tool_call in enumerate(tool_calls):
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_results[i]) if i < len(tool_results) else "Error: No result"
                    })
                
                final_response = call_openai_chat(openai_messages, st.session_state.openai_tools)
                final_content = final_response.content
            else:
                final_content = response_message.content
            
            # Display final response
            message_placeholder.markdown(final_content)
            st.session_state.messages.append({"role": "assistant", "content": final_content})
    
    # Cleanup button
    if st.sidebar.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        st.error("⚠️ Please set OPENAI_API_KEY environment variable")
        st.stop()
    
    main()
