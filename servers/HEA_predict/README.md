# Example MCP Server

This is an example server demonstrating the minimal pattern for creating AI4S MCP tools.

## Key Features

1. **Direct FastMCP usage**: No decorators or abstractions needed
2. **Standard argument parsing**: Consistent CLI interface across all tools
3. **Metadata-driven**: Simple JSON file for tool information
4. **Focus on simplicity**: Write your scientific tools, not boilerplate

## Usage

```bash
# Install dependencies
uv sync

# Run server
python server.py --port 50001

# Run with custom host
python server.py --host localhost --port 50001

# Enable debug logging
python server.py --port 50001 --log-level DEBUG
```

## Creating Your Own Server

1. Copy this example server as a template
2. Update `metadata.json` with your tool's information
3. Implement your scientific functions with `@mcp.tool()`
4. Add dependencies to `pyproject.toml`
5. Run and test your server

## Structure

```
your_tool/
├── server.py         # Main server implementation
├── metadata.json     # Tool metadata for registry
├── pyproject.toml    # Python dependencies
└── README.md         # Tool documentation (optional)
```

## Code Pattern

```python
from fastmcp import FastMCP
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Your tool description")
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()

args = parse_args()
mcp = FastMCP("your-tool", host=args.host, port=args.port)

@mcp.tool()
def your_function(param: str) -> dict:
    """Your function documentation."""
    # Your implementation
    return {"result": "value"}

if __name__ == "__main__":
    mcp.run(transport="sse")
```

## Metadata Format

```json
{
    "name": "YourTool",
    "description": "Brief description of what your tool does",
    "author": "@your-github-username",
    "category": "chemistry"
}
```

Available categories: materials, chemistry, biology, physics, research, simulation, data, machine-learning