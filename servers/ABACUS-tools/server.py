#!/usr/bin/env python3
"""
ABACUS MCP Server - Bridge between AI models and first principles calculations.
Provides tools for ABACUS computational jobs including input preparation, job submission, and analysis.
"""
import sys
from pathlib import Path

# Add parent directory to import server_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from server_utils import mcp_server, setup_server
from mcp.server.fastmcp import FastMCP

# Import ABACUS-specific modules
sys.path.append(str(Path(__file__).resolve().parent))
from src.abacusagent.main import load_tools
from src.abacusagent.env import set_envs


@mcp_server("ABACUS", "First principles calculations bridge for AI models - ABACUS computational jobs", author="@ahxbcn", category="Materials Science")
def create_server(host="0.0.0.0", port=50001):
    """Create and configure the ABACUS MCP server instance."""
    # Set ABACUS environment variables
    set_envs()
    
    # Create MCP server
    mcp = FastMCP("ABACUS", host=host, port=port)
    
    # Load all ABACUS tools (prepare input, modify files, analysis, etc.)
    load_tools()
    
    return mcp


if __name__ == "__main__":
    setup_server().run(transport="sse")