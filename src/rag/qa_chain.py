# 这个文件用于组装“检索 + 生成”的问答链并返回答案与引用来源。
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.rag.prompt import SYSTEM_PROMPT, format_context
from src.rag.retriever import get_retriever


REACT_PLANNER_PROMPT = """
你是金融投研问答中的检索代理。你需要在“检索(retrieve)”和“结束(finish)”之间决策。

请严格按JSON输出，且只能输出一个JSON对象，不要输出其它文字：
{
  "action": "retrieve 或 finish",
  "action_input": "当action=retrieve时填写检索查询词，否则填空字符串",
  "final_answer": "当action=finish时填写最终回答，否则填空字符串"
}

规则：
1) 优先基于已有观察给出答案，避免无效重复检索。
2) 当信息不足时，再执行retrieve并给出更精准查询词。
3) final_answer必须简洁、结构化，且包含“参考来源”段落；若信息不足，明确写“依据当前研报资料无法确定”。
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    # 这个函数用于从模型输出中提取首个JSON对象。
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("planner output does not contain JSON")
    return json.loads(match.group(0))


def _append_unique_docs(pool: list[Document], incoming: list[Document]) -> None:
    # 这个函数用于按来源+页码+内容去重，避免重复片段污染上下文。
    seen = {
        (
            str(doc.metadata.get("source", "unknown")),
            str(doc.metadata.get("page", "?")),
            doc.page_content,
        )
        for doc in pool
    }
    for doc in incoming:
        key = (
            str(doc.metadata.get("source", "unknown")),
            str(doc.metadata.get("page", "?")),
            doc.page_content,
        )
        if key in seen:
            continue
        pool.append(doc)
        seen.add(key)


def _truncate(text: str, max_len: int = 180) -> str:
    # 这个函数用于压缩观察文本长度，控制提示词体积。
    clean = " ".join(text.strip().split())
    if len(clean) <= max_len:
        return clean
    return clean[:max_len] + "..."


def _build_observation(docs: list[Document]) -> str:
    # 这个函数用于把检索结果整理成可供下一轮决策的观察摘要。
    if not docs:
        return "未检索到有效片段。"

    lines: list[str] = []
    for idx, doc in enumerate(docs[:3], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        lines.append(
            f"片段{idx}: {source}(页码:{page}) | 摘要: {_truncate(doc.page_content)}"
        )
    return "\n".join(lines)


def _ask_single_pass(
    llm: ChatOpenAI,
    docs: list[Document],
    question: str,
) -> str:
    # 这个函数用于单轮生成答案，作为ReAct关闭或兜底时的统一出口。
    context = format_context(docs)
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
    return str(result.content)


def _format_sources(docs: list[Document]) -> list[str]:
    sources: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        text = f"{source}（页码: {page}）"
        if text not in sources:
            sources.append(text)
    return sources


def ask_with_rag(settings: Settings, question: str, *, use_react: bool = False) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise EnvironmentError("缺少OPENAI_API_KEY（或DEEPSEEK_API_KEY），请先在.env中配置。")

    retriever = get_retriever(settings)
    docs: list[Document] = retriever.invoke(question)

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    if not use_react:
        answer = _ask_single_pass(llm, docs, question)
        return {
            "answer": answer,
            "sources": _format_sources(docs),
            "retrieved_docs": docs,
            "react_trace": [],
        }

    doc_pool: list[Document] = []
    _append_unique_docs(doc_pool, docs)
    observations: list[str] = [f"初始检索:\n{_build_observation(docs)}"]
    answer: str | None = None

    for step in range(settings.react_max_steps):
        planner_messages = [
            SystemMessage(content=REACT_PLANNER_PROMPT),
            HumanMessage(
                content=(
                    f"用户问题:\n{question}\n\n"
                    f"当前观察:\n{chr(10).join(observations)}\n\n"
                    "请输出下一步JSON决策。"
                )
            ),
        ]
        planner_result = llm.invoke(planner_messages)
        planner_text = str(planner_result.content)

        try:
            plan = _extract_json(planner_text)
        except Exception:
            observations.append(f"第{step + 1}轮规划解析失败，模型原始输出: {_truncate(planner_text, 260)}")
            break

        action = str(plan.get("action", "")).strip().lower()
        action_input = str(plan.get("action_input", "")).strip()
        final_answer = str(plan.get("final_answer", "")).strip()

        if action == "finish" and final_answer:
            answer = final_answer
            observations.append(f"第{step + 1}轮决策: finish")
            break

        query = action_input if action == "retrieve" and action_input else question
        step_docs: list[Document] = retriever.invoke(query)
        _append_unique_docs(doc_pool, step_docs)
        observations.append(
            f"第{step + 1}轮决策: retrieve | query={query}\n观察:\n{_build_observation(step_docs)}"
        )

    if answer is None:
        # 这个分支用于在ReAct未成功finish时回退到稳态单轮生成。
        answer = _ask_single_pass(llm, doc_pool or docs, question)
        observations.append("未在限定步数内finish，已回退单轮RAG生成。")

    return {
        "answer": answer,
        "sources": _format_sources(doc_pool or docs),
        "retrieved_docs": (doc_pool or docs),
        "react_trace": observations,
    }
