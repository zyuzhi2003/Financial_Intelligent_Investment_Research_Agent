# 这个文件用于组装“检索 + 生成”的问答链并返回答案与引用来源。
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.rag.prompt import SYSTEM_PROMPT, format_context
from src.rag.retriever import get_retriever


def _format_sources(docs: list[Document]) -> list[str]:
    sources: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        text = f"{source}（页码: {page}）"
        if text not in sources:
            sources.append(text)
    return sources


def ask_with_rag(settings: Settings, question: str) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise EnvironmentError("缺少OPENAI_API_KEY（或DEEPSEEK_API_KEY），请先在.env中配置。")

    retriever = get_retriever(settings)
    docs: list[Document] = retriever.invoke(question)
    context = format_context(docs)

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "以下是检索到的研报上下文：\n\n"
                f"{context}\n\n"
                "请基于以上内容回答这个问题：\n"
                f"{question}"
            )
        ),
    ]
    result = llm.invoke(messages)

    return {
        "answer": result.content,
        "sources": _format_sources(docs),
        "retrieved_docs": docs,
    }
