# 这个文件用于初始化向量检索器并对外提供相似度检索能力。
from __future__ import annotations

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.embeddings import create_embeddings


def get_retriever(settings: Settings):
    embedding = create_embeddings(settings)
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding,
        persist_directory=str(settings.persist_dir),
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})


def has_index(settings: Settings) -> bool:
    return settings.persist_dir.exists() and any(settings.persist_dir.iterdir())
