#!/usr/bin/env python3
"""
MolPilot服务器 - 整合ORCA工具、 ORCA输出文件解析工具、ORCA手册RAG工具、AutoDE功能等.
"""
import os
import re
import logging
import argparse
import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional, TypedDict, List, Tuple, Dict, Union, Literal, Any
import subprocess
import sys

# from dp.agent.server import CalculationMCPServer
from mcp.server.fastmcp import FastMCP

import autode as ade
import anyio

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
from langchain_core.documents import Document


def parse_args():
    parser = argparse.ArgumentParser(description="Mannual RAG Server")
    parser.add_argument('--port', type=int, default=50008, help='Server port (default: 50008)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50008
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
mcp = FastMCP("mannual_rag_server", host=args.host, port=args.port)

logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s-%(levelname)s-%(message)s"
)


top_n = 3
max_assembly_length = 2000
vector_stores = ["orca_manual"]

embeddings = DashScopeEmbeddings(model="text-embedding-v4")

orca_vector_store = Chroma(
    persist_directory="./vector_db_orca_manual_qwen",
    embedding_function=embeddings
)


class RetrieveContentResult(TypedDict):
    """Retrieve content result."""
    status: str
    retrieved_content: Optional[list[dict[str, Any]]] = None


@mcp.tool()
async def retrieve_content_from_docs(
    query: str,
    vector_store_name: str = "orca_manual",
) -> RetrieveContentResult:
    """
    Retrieve relevant content from documents based on a query.
    
    Args:
        query: The search query
        vector_store_name: Name of the vector store to search in.
         Available options: "orca_manual",
    
    Returns:
        Retrieved content with metadata
    """
    try:

        if vector_store_name not in vector_stores:
            raise ValueError(
                f"Unsupported vector store: {vector_store_name}. Available options: {vector_stores}"
                )


        if vector_store_name == "orca_manual":
            vector_store = orca_vector_store

        final_docs = vector_store.similarity_search(query, k=top_n)
        
        retrieved_content = []
        total_length = 0
        
        for i, doc in enumerate(final_docs):
            content = doc.page_content


            retrieved_content.append({
                "content": content,
                "metadata": doc.metadata,
            })
            
            total_length += len(content)
        
        return RetrieveContentResult(
            status="success",
            retrieved_content=retrieved_content
        )
    except Exception as e:
        logging.error(f"Error retrieving content: {e}")
        return RetrieveContentResult(
            status="error",
            retrieved_content=None
        )


if __name__ == "__main__":
    logging.info("Starting MOLPILOT MCP Server with all tools...")
    mcp.run(transport="sse")
 