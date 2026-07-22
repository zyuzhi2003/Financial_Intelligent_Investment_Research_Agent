# 这个文件用于定义金融RAG问答的系统提示词与上下文拼接逻辑。
from __future__ import annotations

from langchain_core.documents import Document


SYSTEM_PROMPT = """
你是一名严谨的金融投研助手。请严格遵循以下规则：
1) 只能依据提供的上下文回答，不得编造。
2) 若上下文不足，请明确回答“依据当前研报资料无法确定”。
3) 回答要结构化、简洁，并优先给出结论。
4) 回答末尾必须列出“参考来源”，格式为：文件名（页码）。
""".strip()


def format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[片段{idx}]\n来源: {source}\n页码: {page}\n内容:\n{doc.page_content}"
        )
    return "\n\n".join(blocks)
