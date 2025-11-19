"""
Test script for the FE MCP server
"""
import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import subprocess

def send_mcp_request(process, method, params=None):
    """Send a JSON-RPC request to the MCP server."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {}
    }
    
    request_str = json.dumps(request) + "\n"
    process.stdin.write(request_str)
    process.stdin.flush()
    
    response_str = process.stdout.readline()
    return json.loads(response_str)


def test_fe_server():
    """Test the FE server tools."""
    print("="*70)
    print("Testing FE MCP Server")
    print("="*70)
    
    # Launch the server
    print("\n1. Launching server...")
    process = subprocess.Popen(
        [sys.executable, "server_fe.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Initialize
        print("\n2. Initializing...")
        response = send_mcp_request(process, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0.0"}
        })
        print(f"   ✓ Initialized: {response.get('result', {}).get('serverInfo', {}).get('name')}")
        
        # List tools
        print("\n3. Listing tools...")
        response = send_mcp_request(process, "tools/list")
        tools = response.get("result", {}).get("tools", [])
        print(f"   ✓ Found {len(tools)} tools:")
        for tool in tools:
            print(f"     • {tool['name']}: {tool.get('description', '')[:60]}...")
        
        # Test 1: Detect FE files
        print("\n4. Testing fe.detect_files...")
        response = send_mcp_request(process, "tools/call", {
            "name": "fe.detect_files",
            "arguments": {"directory": "."}
        })
        result = json.loads(response["result"]["content"][0]["text"])
        print(f"   Directory: .")
        print(f"   Found in current dir: {result['found']}")
        print(f"   Subdirectories with FE files: {len(result['subdirectories'])}")
        for subdir in result['subdirectories']:
            print(f"     • {subdir['name']}")
        
        # Test 2: Load FE model from 40barTruss
        print("\n5. Testing fe.load_model...")
        response = send_mcp_request(process, "tools/call", {
            "name": "fe.load_model",
            "arguments": {"directory": "40barTruss"}
        })
        result = json.loads(response["result"]["content"][0]["text"])
        print(f"   Success: {result['success']}")
        print(f"   Message: {result['message']}")
        if result['success']:
            print(f"   Nodes: {result['stats']['num_nodes']}")
            print(f"   Elements: {result['stats']['num_elements']}")
        
        # Test 3: Get FE info
        print("\n6. Testing fe.get_info...")
        response = send_mcp_request(process, "tools/call", {
            "name": "fe.get_info",
            "arguments": {"directory": "40barTruss"}
        })
        info = response["result"]["content"][0]["text"]
        print(info)
        
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        process.stdin.close()
        process.terminate()
        process.wait(timeout=5)


if __name__ == "__main__":
    test_fe_server()

