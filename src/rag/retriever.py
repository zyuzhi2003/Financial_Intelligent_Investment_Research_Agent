# 这个文件用于初始化向量检索器并对外提供相似度检索能力。
from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.config import Settings
from src.rag.embeddings import create_embeddings


class _TableAwareRetriever(BaseRetriever):
    vectorstore: Chroma
    top_k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        candidate_count = max(self.top_k * 6, 24)
        vector_candidates = self.vectorstore.similarity_search(query, k=candidate_count)
        stored = self.vectorstore.get(include=["metadatas", "documents"])
        lexical_candidates = [
            Document(page_content=document, metadata=metadata or {})
            for document, metadata in zip(
                stored.get("documents", []), stored.get("metadatas", [])
            )
            if document
        ]
        normalized_query = query.lower()
        normalized_query = re.sub(r"20(\d{2})年", r"\1A", normalized_query)
        normalized_query = normalized_query.replace("上半年", "H1").replace("下半年", "H2")
        aliases = {
            "收入": "营业收入",
            "营收": "营业收入",
            "利润": "净利润",
        }
        terms = [normalized_query]
        terms.extend(re.findall(r"[\u4e00-\u9fff]{2,}|\d{2}q[1-4]|\d{2}h[12]|\d{2}a", normalized_query))
        terms.extend(value for key, value in aliases.items() if key in normalized_query)
        entity_terms = [term for term in terms if len(term) >= 3 and not term.isdigit()]

        def document_key(document: Document) -> tuple[str, str, str]:
            return (
                str(document.metadata.get("source", "")),
                str(document.metadata.get("page", "")),
                document.page_content,
            )

        vector_rank = {
            document_key(document): position
            for position, document in enumerate(vector_candidates)
        }
        merged: dict[tuple[str, str, str], Document] = {
            document_key(document): document for document in vector_candidates
        }
        for document in lexical_candidates:
            merged.setdefault(document_key(document), document)

        def score(document: Document) -> tuple[float, int]:
            searchable = " ".join(
                [
                    document.page_content.lower(),
                    str(document.metadata.get("table_columns", "")).lower(),
                ]
            )
            lexical_hits = sum(1 for term in terms if term and term in searchable)
            table_bonus = 2 if document.metadata.get("chunk_type") == "table" and lexical_hits else 0
            entity_hits = sum(1 for term in entity_terms if term in searchable)
            lexical_score = min(1.0, lexical_hits / max(1, len(set(terms))))
            position = vector_rank.get(document_key(document), candidate_count)
            vector_score = 1.0 / (position + 1) if position < candidate_count else 0.0
            return (
                0.35 * vector_score
                + 0.35 * lexical_score
                + 0.10 * table_bonus
                + 0.20 * min(1.0, entity_hits),
                -position,
            )

        ranked = sorted(merged.values(), key=score, reverse=True)
        return ranked[: self.top_k]


def get_retriever(settings: Settings):
    embedding = create_embeddings(settings)
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding,
        persist_directory=str(settings.persist_dir),
    )
    return _TableAwareRetriever(vectorstore=vectorstore, top_k=settings.top_k)


def has_index(settings: Settings) -> bool:
    return settings.persist_dir.exists() and any(settings.persist_dir.iterdir())
