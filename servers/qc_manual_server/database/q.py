from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma(persist_directory="./vector_db", embedding_function=embedding)

# class RetrievalTool:
#     def __init__(self, vectorstore):
#         self.vectorstore = vectorstore

#     def __call__(self, query: str, top_k: int = 3):
#         """检索向量存储并返回相关文档内容"""
#         try:
#             docs = self.vectorstore.similarity_search(query, k=top_k)
#             return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
#         except Exception as e:
#             return f"检索失败: {str(e)}"

# retrieval_tool = RetrievalTool(vectorstore)
# query = "VCD input file"  # 替换为你的查询
# result = retrieval_tool(query)
# print(result)
def retrieve_content(query: str) -> dict:
    try:
        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(
            persist_directory="./vector_db", 
            # persist_directory="./database/vector_db", 
            embedding_function=embedding
        )
        retrieved_docs = vector_store.similarity_search_with_score(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata['source']} {doc.metadata['page_label']}/{doc.metadata['total_pages']}\n" f"Score: {score}\n" f"Content:\n{doc.page_content}")
            for doc, score in retrieved_docs
        )
        # print(serialized)
        return {
            "status": "success",
            "message": "Content retrieved successfully.",
            "retrieved_content": serialized,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve content: {str(e)}",
        }
# print(retrieve_content("Multiwfn manual section 3.13.3"))
print(retrieve_content("first input path of output file of aforementioned programs or the plain text file containing transition data, then enter main function 11, and you will be asked to select the type of spectrum"))
