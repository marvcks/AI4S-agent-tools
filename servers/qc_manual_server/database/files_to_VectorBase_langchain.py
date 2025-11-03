from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredMarkdownLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter


def add_pdf_to_vector_db(pdf_path: str, vector_store: Chroma, source: str):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(docs)}")

    for doc in docs:
        doc.metadata['source'] = source

    vector_store.add_documents(docs)
    # vectorstore.persist()


def add_md_to_verctor_db(md_path: str, vector_store: Chroma, source: str):
    """
    Add markdown files to the vector database.
    
    Args:
        md_path (str): Path to the markdown file.
        vector_store (Chroma): Vector store instance.
    """

    loader = UnstructuredMarkdownLoader(md_path)
    documents = loader.load()
    documents[0].metadata['source'] = source

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    # print(f"Total chunks created: {len(docs)}")
    # docs = documents

    vector_store.add_documents(docs)
    # vector_store.add_documents(documents)


if __name__ == "__main__":
    from pathlib import Path

    embedding = DashScopeEmbeddings(model="text-embedding-v2")
    vector_store = Chroma(
        persist_directory="/Users/xiaohuxu/Documents/python/deepmodeling/OrcaMul/servers/multiwfn/database/vector_db_qwen",
        embedding_function=embedding
    )

    md_path = Path("/Users/xiaohuxu/Documents/python/sob_blobs/sobereva_blogs_text")
    md_files = list(md_path.glob("*.md"))
    for md_file in md_files:
        add_md_to_verctor_db(
            md_path=str(md_file),
            vector_store=vector_store,
            source="sobereva_blog"
        )
    
    # test for add one md
    # add_md_to_verctor_db(
    #     md_path="/Users/xiaohuxu/Documents/python/sob_blobs/sobereva_blogs_text/227.md",
    #     vector_store=vector_store,
    #     source="sobereva_blog"
    # )

    # # test for add one pdf
    # add_pdf_to_vector_db(
    #     pdf_path="/Users/xiaohuxu/Documents/python/deepmodeling/OrcaMul/servers/multiwfn/database/Multiwfn_3.8_dev.pdf",
    #     vector_store=vector_store,
    #     source="multiwfn_doc"
    # )
    
    # g16_doc = "/Users/xiaohuxu/Documents/python/deepmodeling/OrcaMul/servers/gaussian16/Gaussian16UsersReference.pdf"
    # add_pdf_to_vector_db(
    #     pdf_path=g16_doc,
    #     vector_store=vector_store,
    #     source="Gaussian16UserReference.pdf"
    # )

