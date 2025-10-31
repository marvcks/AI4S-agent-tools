# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
from typing import Any, Optional, TypedDict
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

embedding = DashScopeEmbeddings(model="text-embedding-v2")
reranker = DashScopeRerank(model="gte-rerank")

vector_store = Chroma(
    persist_directory="./vector_db_qwen",
    embedding_function=embedding
)

class RetrieveContentResult(TypedDict):
    """Retrieve content result."""
    status: str
    message: str
    retrieved_content: Optional[list[dict[str, Any]]] = None


# @mcp.tool()
async def retrieve_content(query: str, top_n: int, source: Optional[str] = None) -> RetrieveContentResult:
    """
    Retrieve content based on the query from Multiwfn docs and blogs.
    
    Args:
        query (str): 
            The query to search for.
        top_n (int): 
            The number of top results to return.
        source (Optional[str]): 
            The source to filter results by. 
            Must be either "Multiwfn_3.8_dev.pdf" or "sobereva_blog" or "Gaussian16UserReference.pdf". 
            If None, no filtering is applied.

    Returns:
        dict: 
            A dictionary containing the status, message, and retrieved content.
    """
    try:
        if not source:
            ifilter = None
        elif source not in ["Multiwfn_3.8_dev.pdf", "sobereva_blog", "Gaussian16UserReference.pdf"]:
            return RetrieveContentResult(
                status="error",
                message="Invalid source specified. Use 'Multiwfn_3.8_dev.pdf' or 'sobereva_blog' or 'Gaussian16UserReference.pdf' .",
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

if __name__ == "__main__":
    import asyncio

    # Example usage
    query = "VCD input file"  # Replace with your query
    # result = asyncio.run(retrieve_content(query, 1))
    # result = asyncio.run(retrieve_content(query, 1, "Multiwfn_3.8_dev.pdf"))
    result = asyncio.run(retrieve_content(query, 1, "Gaussian16UserReference.pdf"))
    print(result)