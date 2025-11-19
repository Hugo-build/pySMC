"""
Direct test of FE detection functions (without MCP protocol)
"""
import sys
from pathlib import Path

# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the functions directly from server_fe
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the tool functions by executing the module
import importlib.util
spec = importlib.util.spec_from_file_location("server_fe", "server_fe.py")
server_fe = importlib.util.module_from_spec(spec)

# Mock the FastMCP decorator to extract the actual functions
original_tools = {}

class MockFastMCP:
    def __init__(self, name):
        self.name = name
    
    def tool(self, name=None):
        def decorator(func):
            original_tools[name or func.__name__] = func
            return func
        return decorator
    
    def run(self):
        pass

# Replace FastMCP in server_fe module
import fastmcp
original_fastmcp = fastmcp.FastMCP
fastmcp.FastMCP = MockFastMCP

# Now load the module
spec.loader.exec_module(server_fe)

# Restore original FastMCP
fastmcp.FastMCP = original_fastmcp

# Get the functions
detect_fe_files = original_tools["fe.detect_files"]
load_fe_model = original_tools["fe.load_model"]
get_fe_info = original_tools["fe.get_info"]

# ============================================================================
# Run Tests
# ============================================================================

def test_fe_functions():
    """Test the FE functions directly."""
    print("="*70)
    print("Testing FE Functions (Direct)")
    print("="*70)
    
    # Test 1: Detect files in current directory
    print("\n1. Testing detect_fe_files('.')...")
    result = detect_fe_files(".")
    print(f"   Found in current dir: {result['found']}")
    print(f"   Subdirectories: {len(result['subdirectories'])}")
    for subdir in result['subdirectories']:
        print(f"     • {subdir['name']}: {subdir['path']}")
    print(f"   Message: {result['message']}")
    
    # Test 2: Detect files in 40barTruss
    print("\n2. Testing detect_fe_files('40barTruss')...")
    result = detect_fe_files("40barTruss")
    print(f"   Found: {result['found']}")
    print(f"   Nodes file: {result['nodes_file']}")
    print(f"   Elements file: {result['elements_file']}")
    print(f"   Message: {result['message']}")
    
    # Test 3: Load FE model
    print("\n3. Testing load_fe_model('40barTruss')...")
    result = load_fe_model("40barTruss")
    print(f"   Success: {result['success']}")
    print(f"   Message: {result['message']}")
    if result['success']:
        print(f"   Nodes: {result['stats']['num_nodes']}")
        print(f"   Elements: {result['stats']['num_elements']}")
        print(f"   First node: {result['nodes'][0]}")
        print(f"   First element: {result['elements'][0]}")
    
    # Test 4: Get info
    print("\n4. Testing get_fe_info('40barTruss')...")
    info = get_fe_info("40barTruss")
    print(info)
    
    print("\n" + "="*70)
    print("✓ All tests completed!")
    print("="*70)


if __name__ == "__main__":
    test_fe_functions()

