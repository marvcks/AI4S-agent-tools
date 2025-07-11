import sys
from pathlib import Path

# Add parent directory to import server_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from server_utils import mcp_server, setup_server

from mcp.server.fastmcp import FastMCP
from python_version.mcp_server import handle_tool_call


@mcp_server("pubchem", "PubChem compound data retrieval", author="@NyankoDev", category="chemistry")
def create_server(host="0.0.0.0", port=50001):
    """Create and configure the MCP server instance."""
    mcp = FastMCP("pubchem", host=host, port=port)

    @mcp.tool()
    def get_pubchem_data(query: str, format: str = 'JSON', include_3d: bool = False) -> str:
        """
        Get PubChem data for a given query.
        """
        return handle_tool_call("get_pubchem_data", {"query": query, "format": format, "include_3d": include_3d})

    @mcp.tool()
    def download_structure(cid: str, format: str = 'sdf') -> str:
        """
        Download a structure from PubChem.
        """
        return handle_tool_call("download_structure", {"cid": cid, "format": format})

    return mcp

if __name__ == "__main__":
    setup_server().run("sse")



