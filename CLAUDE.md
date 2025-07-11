# CLAUDE.md
- 用中文回答用户的所有问题
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI4S-agent-tools is an open project by the DeepModeling community focused on collecting agentic tools for scientific research. It builds a "scientific capability library" that can be invoked by intelligent agents, implemented as MCP (Model Context Protocol) servers.

## Architecture

The repository follows a modular architecture where each scientific tool is implemented as an independent MCP server under the `/servers/` directory. Each server:

- Has its own `pyproject.toml` and `uv.lock` for dependency management
- Exposes tools via the MCP protocol using either FastMCP or CalculationMCPServer frameworks
- Runs on a unique port (see TOOLS.json for port allocations)
- Uses Server-Sent Events (SSE) or streamable-http as transport protocol

### Tool Registry

All available tools are automatically cataloged in `TOOLS.json`. To regenerate this file:

```bash
python scripts/generate_tools_json.py
```

### Key Servers

- **DPACalculator**: Deep Potential Agent for atomistic simulations (uses science-agent-sdk)
- **pubchem**: PubChem compound data retrieval (has both FastMCP and Python MCP implementations)
- **Paper_Search**: ArXiv paper search functionality
- **deepmd_docs_rag**: RAG-based documentation server for DeepMD
- **thermoelectric_materials_mcp_server**: Thermoelectric materials database access
- **catalysis**: Catalysis reaction calculations based on ADSEC workflow

## Development Commands

### Dependency Management

The project uses UV for Python dependency management. To install dependencies:

```bash
# Install UV if not already installed
pip install uv

# Install dependencies for a specific server
cd servers/<server_name>
uv sync
```

### Running Servers

Each server supports command-line arguments for flexible configuration:

```bash
# Run server (port is required)
cd servers/<server_name>
python server.py --port 50001

# Run with custom host
python server.py --host localhost --port 51000

# Enable debug logging
python server.py --port 50001 --log-level DEBUG

# View help
python server.py --help
```


### Error Handling Pattern

Always wrap tool implementations with try-except blocks:

```python
try:
    # Main logic
    return {"result": value}
except Exception as e:
    logging.error(f"Error: {str(e)}", exc_info=True)
    return {"error": f"Failed: {str(e)}"}
```

## Code Patterns

### Minimal Server Pattern (Recommended)

```python
from servers.server_utils import mcp_server, setup_server
from mcp.server.fastmcp import FastMCP

@mcp_server("YourToolName", "Clear description of your scientific tool", author="@your-github", category="materials")
def create_server(host="0.0.0.0", port=50001):
    mcp = FastMCP("your_tool", host=host, port=port)
    
    @mcp.tool()
    def your_function(param: str) -> dict:
        """Function documentation."""
        return {"result": "value"}
    
    return mcp

if __name__ == "__main__":
    setup_server().run()  # One line!
```

Benefits:
- Minimal decorator - only 3 required parameters (name, description, author)
- Port specified at runtime with --port argument
- Auto-detection of decorated function
- TOOLS.json auto-generation support

## Testing

Currently, the project lacks a formal testing framework. When adding new functionality:

1. Test servers manually by running them and calling their tools
2. Check server logs for errors (typically in `~/.{server-name}/` directory)
3. Verify MCP protocol compliance using the client integration

## Important Notes

- **Shared utilities**: Use `servers/server_utils.py` for common server functionality (@mcp_server decorator, setup_server)
- **Python version**: All servers require Python >= 3.8
- **Dependencies**: Always add new dependencies to the appropriate `pyproject.toml` file
- **Port**: Specified at runtime with --port argument
- **Logging**: Configured via --log-level argument (default: INFO)
- **Author attribution**: Include your GitHub handle or team name in @mcp_server decorator
- **Category**: Specify category in @mcp_server decorator using human-readable names (e.g., "Materials Science", "Chemistry"). The generate_tools_json.py script will map these to category keys

## Adding a New Server

### Quick Start (Recommended)

1. Copy the example server as a template:
   ```bash
   cp -r servers/_example servers/your_tool_name
   ```

2. Update `server.py` with your tool's information:
   - Modify the `@mcp_server` decorator with your metadata
   - Implement your scientific functions with `@mcp.tool()`
   - Update the default port to a unique value

3. Update `pyproject.toml` with your dependencies

4. Add a README.md explaining your tool's functionality

5. Regenerate TOOLS.json:
   ```bash
   python scripts/generate_tools_json.py
   ```

### Manual Steps (if not using template)

1. Create a new directory under `/servers/`
2. Add a `pyproject.toml` with appropriate dependencies
3. Create `server.py` using the minimal @mcp_server decorator pattern
4. Choose a unique port number (check TOOLS.json for used ports)
5. Add documentation in the server's README
6. Run the generation script to update TOOLS.json

## MCP Client Integration

Servers are integrated with MCP clients (like Claude Desktop) by updating the client's configuration file with:

```json
{
  "mcpServers": {
    "server_name": {
      "command": "python3",
      "args": ["/full/path/to/server.py"],
      "env": {"PYTHONUNBUFFERED": "1"},
      "disabled": false
    }
  }
}
```

## Showcase Page

The project includes an auto-generated showcase page that displays all available tools organized by category. 

### Generating the Showcase

```bash
# Generate the showcase page manually
python scripts/generate_simple_showcase.py
```

The showcase page is automatically regenerated and deployed via GitHub Actions when:
- TOOLS.json is updated
- categories.json is modified
- The generation scripts are changed

### Category Mapping

When specifying categories in the @mcp_server decorator, use human-readable names:
- "Materials Science" → materials
- "Chemistry" → chemistry
- "Biology" → biology
- "Physics" → physics
- "Research Tools" → research
- "Simulation" → simulation
- "Data & Analysis" → data
- "Machine Learning" → machine-learning
- "General Tools" → general

The `generate_tools_json.py` script automatically maps these human-readable names to the category keys defined in `config/categories.json`.