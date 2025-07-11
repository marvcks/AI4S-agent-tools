
import sys
from pathlib import Path

# Add parent directory to import server_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from server_utils import mcp_server, setup_server

from fastmcp import FastMCP, Client
from typing import List
import requests
import json
import os
from dotenv import load_dotenv


load_dotenv()
FASTGPT_AUTH_TOKEN = os.getenv("FASTGPT_AUTH_TOKEN")


@mcp_server("deepmd_docs_rag", "DeepMD documentation RAG server", author="@felix5572", category="Materials Science")
def create_server(host="0.0.0.0", port=50001):
    """Create and configure the MCP server instance."""
    backend = Client("https://api.fastgpt.in/api/mcp/app/wrSvRtDydWEb7faAP6uPooSX/mcp")
    mcp = FastMCP.as_proxy(backend, name="FastGPT Proxy Server", host=host, port=port)
    @mcp.tool()
    def upload_single_file_to_deepmd_knowledge_base(file_url: str):
        """get the file from the specified URL and upload it DeePMD-docs knowledge database.
        These files will be used to train the DeePMD-docs knowledge base.
        And future chat will use these documents to answer questions.
        """

        API_URL = "https://api.fastgpt.in/api/core/dataset/collection/create/localFile"
        DATASET_ID = "683a517f292645ebd7a2cb7c"      #  knowledge base: deepmd-from-chat

        # 1. download the file from the specified URL
        file_name = file_url.split("/")[-1].replace(" ", "_")
        file_content = requests.get(file_url).content
        
        # 2. upload the file to the specified dataset
        response = requests.post(
            url=API_URL,
            headers={"Authorization": f"Bearer {FASTGPT_AUTH_TOKEN}"},
            files={
                'file': (file_name, file_content),
                'data': (None, json.dumps({"datasetId": DATASET_ID, "parentId": None, "trainingType": "chunk", "chunkSize": 512}))
            }
        )
        
        print(f"Status: {response.status_code}, Response: {response.text}")
        return response

    return mcp


if __name__ == "__main__":
    # mcp.run(transport='sse')
    setup_server().run(transport="sse")
