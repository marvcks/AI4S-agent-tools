#!/usr/bin/env python3
"""
Example MCP Server using the minimal @mcp_server decorator pattern.
This demonstrates how to create a new AI4S tool with minimal code.
"""
import sys
from pathlib import Path

# Add parent directory to import server_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from server_utils import mcp_server, setup_server
from mcp.server.fastmcp import FastMCP


@mcp_server("ExampleTool", "Example AI4S tool demonstrating the minimal pattern", author="@example-author", category="General Tools")
def create_server(host="0.0.0.0", port=50001):
    """Create and configure the MCP server instance."""
    mcp = FastMCP("example", host=host, port=port) #or CalculationMCPServer("example", host=host, port=port)
    
    @mcp.tool()
    def example_calculation(x: float, y: float) -> dict:
        """
        Perform an example calculation.
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Dictionary with sum, product, and ratio
        """
        return {
            "sum": x + y,
            "product": x * y,
            "ratio": x / y if y != 0 else "undefined"
        }
    
    @mcp.tool()
    def demo_function(message: str) -> str:
        """
        A demo function that echoes the input message.
        
        Args:
            message: The message to echo
            
        Returns:
            The echoed message with a prefix
        """
        return f"Echo: {message}"
    
    return mcp


if __name__ == "__main__":
    setup_server().run(transport="sse")