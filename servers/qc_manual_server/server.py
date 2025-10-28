import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from typing import Any, Optional, TypedDict

from mcp.server.fastmcp import FastMCP

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    )

logging.info("Embedding model and reranker initialized.")
embedding = DashScopeEmbeddings(model="text-embedding-v2")
reranker = DashScopeRerank(model="gte-rerank")

logging.info("Vector store initialized.")
vector_store = Chroma(
    persist_directory="./database/vector_db_qwen",
    embedding_function=embedding
)

sob_blobs_dir = Path("/Users/xhxu/Documents/AI4S-agent-tools/servers/qc_manual_server/sobereva_blogs_text")

mcp = FastMCP(
    "multiwfn_server",
    host="0.0.0.0", port=50001
    )


class RetrieveContentResult(TypedDict):
    """Retrieve content result."""
    status: str
    message: str
    retrieved_content: Optional[list[dict[str, Any]]] = None


@mcp.tool()
async def retrieve_content(query: str, top_n: int, source: Optional[str] = None) -> RetrieveContentResult:
    """
    Retrieve content based on the query.
    
    Args:
        query (str): 
            The query to search for.
        top_n (int): 
            The number of top results to return.
        source (Optional[str]): 
            The source to filter results by. 
            Must be either "Multiwfn_3.8_dev.pdf" or "Gaussian16UserReference.pdf". 
            If None, no filtering is applied.

    Returns:
        dict: 
            A dictionary containing the status, message, and retrieved content.
    """
    try:
        if not source:
            ifilter = None
        elif source not in ["Multiwfn_3.8_dev.pdf", "Gaussian16UserReference.pdf"]:
            return RetrieveContentResult(
                status="error",
                message="Invalid source specified. Use 'Multiwfn_3.8_dev.pdf' or 'Gaussian16UserReference.pdf' .",
                retrieved_content=None
            )
        else:
            ifilter = {"source": source}

        retrieved_docs = vector_store.similarity_search(query, k=top_n*3, filter=ifilter)
        if not retrieved_docs:
            return RetrieveContentResult(
                status="error",
                message="No content found for the given query.",
                retrieved_content=None
            )
        reranked_index = reranker.rerank(
            query=query,
            documents=retrieved_docs,
            top_n=top_n
        )
        # reranked_index:
        # [
        #   {'index': 6, 'relevance_score': 0.588898802542173}, 
        #   {'index': 7, 'relevance_score': 0.28424317977467733}, 
        #   {'index': 8, 'relevance_score': 0.25712842689333487}
        # ]
        reranked_docs = [retrieved_docs[i['index']] for i in reranked_index]
        
        serialized = [
            {
                "content": doc.page_content,
                # "metadata": doc.metadata,
            }
            for doc in reranked_docs
        ]
        return RetrieveContentResult(
            status="success",
            message="Content retrieved successfully.",
            retrieved_content=serialized
        )
    except Exception as e:
        return RetrieveContentResult(
            status="error",
            message=f"Failed to retrieve content: {str(e)}",
            retrieved_content=None
        )


class SobBlogRetrieveResult(TypedDict):
    """Retrieve content result."""
    status: str
    content: Optional[str] = None


@mcp.tool()
def list_sobereva_blogs() -> dict[str, str]:
    """
    Return a mapping of sobereva blog IDs to their titles.

    Returns:
            dict[str, str]: Mapping from markdown filename stems to processed first-line strings.

    Example:
            # The returned dict will include: 
            # {'57': '#Gaussian的Link、IOp与非标准计算路径', '512': '#使用Multiwfn对第一超极化率做双能级和三能级模型分析', ...}
    """
    blog_ids = {p.stem: p.read_text().split("\n")[0].replace(" ", "") for p in sob_blobs_dir.glob("*.md")}
    return blog_ids


@mcp.tool()
def get_sobereva_blog(blog_id: int) -> SobBlogRetrieveResult:
    """
    Retrieve sobereva blog by its numeric identifier.

    Parameters
    ----------
    blog_id : int
        Numeric identifier of the blog post.

    Returns
    -------
    SobBlogRetrieveResult
        A mapping with at least the following keys:
          - "status": "success" when the blog is successfully read.
          - "content": the blog contents.
    Example Output: {'status': 'success', 'content': '#使用NICS和磁感生电流考察...'}

    """
    blog_id = str(blog_id)
    blog_content = sob_blobs_dir / f"{blog_id}.md"
    content = blog_content.read_text().strip().replace(" ", "").replace("\n\n", "")
    return SobBlogRetrieveResult(
        status="success", 
        content=content
    )

if __name__ == "__main__":
    logging.info("Starting RAG MCP Server...")
    mcp.run(transport='sse')
    # mcp.run(transport='streamable-http')
