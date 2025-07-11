# Example MCP Server

This is an example server demonstrating the minimal pattern for creating AI4S MCP tools.

## Key Features of the Minimal Pattern

1. **@mcp_server decorator**: Only 3 required parameters (name, description, author)
2. **setup_server()**: Auto-detects decorated function and handles arguments
3. **One-line execution**: `setup_server().run(transport="sse")`
4. **Port is specified at runtime**: Use `--port` argument when starting

## Usage

```bash
# Run server (port is required)
python server.py --port 50001

# Run with custom host
python server.py --host localhost --port 51000

# Enable debug logging
python server.py --port 50001 --log-level DEBUG
```

## Creating Your Own Server

1. Copy this example server as a template
2. Update the `@mcp_server` decorator with your tool's information
3. Implement your scientific functions with `@mcp.tool()`
4. That's it!

## Benefits

- **Minimal code**: Just what you need, nothing more
- **Runtime port configuration**: Specify port when starting the server
- **Auto-generated TOOLS.json**: Run `python scripts/generate_tools_json.py`
- **Focus on science**: Write your tools, not boilerplate

## Minimal Code Structure

```python
@mcp_server("YourTool", "Your tool description", author="@your-github-handle")
def create_server(host="0.0.0.0", port=50001):
    mcp = FastMCP("your_tool", host=host, port=port)
    
    @mcp.tool()
    def your_function(param: type) -> dict:
        """Your function documentation."""
        # Your implementation
        return result
    
    return mcp

if __name__ == "__main__":
    setup_server().run()  # That's it!
```