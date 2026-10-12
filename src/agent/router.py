# 这个文件用于实现阶段四的统一执行路由与结果编排。
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.intent import IntentResult, RouteType, classify_intent
from src.config import Settings
from src.memory import ChatMemoryStore, HybridMemoryStore, MemoryScope
from src.mcp import MCPDispatcher
from src.mcp.nl_router import build_request_from_question
from src.rag.qa_chain import ask_with_rag
from src.rag.prompt import format_context
from src.rag.retriever import get_retriever


AGENT_REACT_PLANNER_PROMPT = """
你是金融投研Agent的ReAct编排器。
你必须在每一步先给出Thought，再给出Action。

仅允许以下Action：
1) rag_retrieve: 检索研报片段，action_input为检索查询词。
2) mcp_call: 调用MCP工具计算，action_input为自然语言计算指令。
3) finish: 结束并输出最终答案。

请严格输出一个JSON对象，不要输出其它文字：
{
    "thought": "本步思考",
    "action": "rag_retrieve | mcp_call | finish",
    "action_input": "动作输入，finish时可为空",
    "final_answer": "仅当action=finish时填写最终回答，否则留空"
}

约束：
1) 若已有足够观察信息，优先finish，避免无效循环。
2) 如果用户问题包含明确数值测算需求，必须至少执行一次mcp_call。
3) 若信息不足，再执行rag_retrieve。
4) 回答中禁止编造；信息不足时明确说明“依据当前资料无法确定”。
""".strip()


AGENT_REACT_SYNTHESIS_PROMPT = """
你是严谨的金融投研助手。
请基于提供的ReAct观察结果给出最终回答：
1) 先结论，后依据；
2) 仅基于观察内容，不得编造；
3) 若信息不足，明确写“依据当前资料无法确定”；
4) 若有MCP计算结果，请解释其含义；
5) 回答末尾添加“参考来源”小节（若无来源可写“无”）。
""".strip()


@dataclass(frozen=True)
class RouteOutput:
    # 这个字段用于返回路由类型。
    route: RouteType
    # 这个字段用于返回可直接展示给用户的答案。
    answer: str
    # 这个字段用于返回路由解释与执行过程。
    trace: list[str]
    # 这个字段用于返回引用来源。
    sources: list[str]
    # 这个字段用于返回机器可读扩展数据。
    data: dict[str, Any]


class AgentRouter:
    # 这个类用于统一编排RAG、MCP和联合推理调用链。
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mcp = MCPDispatcher()
        self._memory: HybridMemoryStore | None = None
        self._chat_memory: ChatMemoryStore | None = None
        self._profile_candidate_min_confidence = 0.45

    def _get_chat_memory(self) -> ChatMemoryStore:
        # 这个函数用于仅在chat实际调用时惰性创建SQLite聊天存储。
        if self._chat_memory is None:
            self._chat_memory = ChatMemoryStore(self._settings.memory_dir)
        return self._chat_memory

    def _build_memory_store(self) -> HybridMemoryStore | None:
        # 这个函数用于按配置创建记忆存储实例。
        if not self._settings.memory_enabled:
            return None
        try:
            return HybridMemoryStore(self._settings)
        except Exception:  # noqa: BLE001
            return None

    def route_only(self, question: str) -> IntentResult:
        # 这个方法用于仅返回意图识别结果。
        return classify_intent(question, self._settings)

    def ask(self, question: str, *, user_id: str = "", session_id: str = "") -> RouteOutput:
        # 这个方法用于执行无状态的单轮自动分流，兼容参数不会启用聊天记忆。
        _ = (user_id, session_id)
        intent = classify_intent(question, self._settings)
        intent = self._stabilize_intent(question, intent)

        if self._settings.enable_react:
            # 这个分支用于优先启用Agent级ReAct编排；失败时回退原有路径。
            try:
                return self._handle_agent_level_react(question, intent)
            except Exception:  # noqa: BLE001
                pass

        return self._dispatch_legacy(question, intent)

    def chat(self, question: str, *, user_id: str, session_id: str) -> RouteOutput:
        # 这个方法用于加载标准消息历史并执行有状态聊天问答。
        chat_memory = self._get_chat_memory()
        session = chat_memory.ensure_session(session_id=session_id, user_id=user_id)
        stored_messages = chat_memory.list_messages(
            session_id=session.session_id,
            limit=self._settings.chat_history_limit,
        )
        conversation_messages: list[BaseMessage] = []
        profile = chat_memory.get_profile(user_id=user_id)
        if profile is not None:
            conversation_messages.append(
                SystemMessage(content=f"用户长期偏好：{json.dumps(profile.preferences, ensure_ascii=False)}")
            )
        for message in stored_messages:
            if message.role == "user":
                conversation_messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                conversation_messages.append(AIMessage(content=message.content))

        chat_memory.append_message(
            session_id=session.session_id,
            user_id=user_id,
            role="user",
            content=question,
        )
        intent = self._stabilize_intent(question, classify_intent(question, self._settings))
        try:
            if self._settings.enable_react:
                try:
                    output = self._handle_agent_level_react(
                        question,
                        intent,
                        conversation_messages=conversation_messages,
                    )
                except Exception:  # noqa: BLE001
                    output = self._dispatch_legacy(
                        question,
                        intent,
                        conversation_messages=conversation_messages,
                    )
            else:
                output = self._dispatch_legacy(
                    question,
                    intent,
                    conversation_messages=conversation_messages,
                )
        except Exception as exc:  # noqa: BLE001
            output = RouteOutput(
                intent.route,
                "当前请求暂时无法执行，请确认向量索引、模型服务和 API 配置后重试。",
                [intent.reason, f"执行失败：{type(exc).__name__}"],
                [],
                {"error_type": type(exc).__name__},
            )

        chat_memory.append_message(
            session_id=session.session_id,
            user_id=user_id,
            role="assistant",
            content=output.answer,
            payload={"route": output.route.value, "sources": output.sources},
        )
        profile_status = self._update_chat_profile(question=question, answer=output.answer, user_id=user_id)
        data = dict(output.data)
        data["chat_memory"] = {
            "session_id": session.session_id,
            "loaded_messages": len(stored_messages),
            "profile_loaded": profile is not None,
            "profile_write_status": profile_status,
            "write_status": "ok",
        }
        return RouteOutput(output.route, output.answer, output.trace, output.sources, data)

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        # 这个函数用于复用关键词匹配逻辑。
        return any(keyword in text for keyword in keywords)

    def _stabilize_intent(self, question: str, intent: IntentResult) -> IntentResult:
        # 这个函数用于对LLM路由结果做硬规则纠偏，降低mcp_only与rag_mcp漂移。
        raw = question.strip()
        lower = raw.lower()

        has_numeric = any(char.isdigit() for char in raw)
        has_metric = self._contains_any(lower, ["pe", "市盈率", "资产负债率"])
        has_calc = self._contains_any(raw, ["计算", "测算", "估值", "估值高低", "高估", "低估", "判断估值", "估值水平"])
        has_research = self._contains_any(
            raw,
            ["结合研报", "研报", "行业", "同业", "平均", "基准", "中枢", "区间", "对标", "政策", "走势"],
        )

        target_route: RouteType | None = None
        reason = ""
        if has_metric and has_numeric and has_research:
            target_route = RouteType.RAG_MCP
            reason = "检测到行业/研报对标与数值测算并存，固定走rag_mcp。"
        elif has_metric and has_numeric and has_calc:
            target_route = RouteType.MCP_ONLY
            reason = "检测到指标测算且无检索信号，固定走mcp_only。"

        if target_route is None or target_route == intent.route:
            return intent

        return IntentResult(
            route=target_route,
            reason=f"{intent.reason} | 路由稳定化修正：{reason}",
        )

    def _dispatch_legacy(
        self,
        question: str,
        intent: IntentResult,
        *,
        conversation_messages: list[BaseMessage] | None = None,
    ) -> RouteOutput:
        # 这个方法用于保留原有稳定路由逻辑作为兜底。
        if intent.route == RouteType.MCP_ONLY:
            return self._handle_mcp_only(question, intent)

        if intent.route == RouteType.RAG_ONLY:
            return self._handle_rag_only(question, intent, conversation_messages=conversation_messages)

        if intent.route == RouteType.RAG_MCP:
            return self._handle_rag_mcp(question, intent, conversation_messages=conversation_messages)

        return self._handle_general(question, intent, conversation_messages=conversation_messages)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        # 这个函数用于从模型输出中提取首个JSON对象。
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("planner output does not contain JSON")
        return json.loads(match.group(0))

    @staticmethod
    def _truncate(text: str, max_len: int = 180) -> str:
        # 这个函数用于控制观察文本长度，避免轨迹过长。
        clean = " ".join(str(text).strip().split())
        if len(clean) <= max_len:
            return clean
        return clean[:max_len] + "..."

    @staticmethod
    def _append_unique_docs(pool: list[Document], incoming: list[Document]) -> None:
        # 这个函数用于按来源+页码+内容去重，防止重复检索片段膨胀。
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

    @staticmethod
    def _format_sources(docs: list[Document]) -> list[str]:
        # 这个函数用于把文档元信息整理为去重来源列表。
        sources: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            text = f"{source}（页码: {page}）"
            if text not in sources:
                sources.append(text)
        return sources

    def _build_rag_observation(self, docs: list[Document]) -> str:
        # 这个函数用于生成检索动作的Observation文本。
        if not docs:
            return "未检索到有效片段。"
        lines: list[str] = []
        for idx, doc in enumerate(docs[:2], start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            if doc.metadata.get("chunk_type") == "table":
                rows = doc.page_content.splitlines()
                columns = str(doc.metadata.get("table_columns", ""))
                first_row = rows[1] if len(rows) > 1 else ""
                summary = self._truncate(
                    f"表头: {columns}\n首行数据: {first_row}",
                    max_len=520,
                )
            else:
                summary = self._truncate(doc.page_content)
            lines.append(f"片段{idx}: {source}(页码:{page}) | 摘要: {summary}")
        return "\n".join(lines)

    def _build_mcp_observation(self, response: dict[str, Any]) -> str:
        # 这个函数用于生成计算动作的Observation文本。
        if not response.get("ok", False):
            error = response.get("error", {}) if isinstance(response.get("error"), dict) else {}
            return f"MCP调用失败: {error.get('message', '未知错误')}"
        result = response.get("result", {}) if isinstance(response.get("result"), dict) else {}
        formula = str(result.get("formula", "无公式"))
        interpretation = str(result.get("interpretation", "无解读"))
        return f"MCP结果: {formula} | 解读: {interpretation}"

    @staticmethod
    def _with_memory_context(question: str, memory_context: str) -> str:
        # 这个函数用于把记忆上下文拼接到问题中供下游链路使用。
        if not memory_context.strip():
            return question
        return f"{question}\n\n[记忆上下文]\n{memory_context}"

    def _format_memory_records(self, records: list[Any]) -> str:
        # 这个函数用于把命中记忆压缩为可注入的简洁文本。
        lines: list[str] = []
        for idx, record in enumerate(records, start=1):
            lines.append(f"{idx}. ({record.scope.value}) {self._truncate(record.text, max_len=140)}")
        return "\n".join(lines)

    def _load_memory_context(
        self,
        *,
        question: str,
        user_id: str,
        session_id: str,
    ) -> tuple[str, dict[str, Any]]:
        # 这个函数用于汇总三类记忆并返回上下文与统计信息。
        if self._memory is None:
            return "", {"recalled_count": 0, "scope_counts": {}, "enabled": False}

        scope_counts = {"session": 0, "user": 0, "project": 0}
        recalled_records: list[Any] = []

        if session_id.strip():
            session_records = self._memory.recall(
                query=question,
                scope=MemoryScope.SESSION,
                top_k=self._settings.memory_top_k,
                user_id=user_id,
                session_id=session_id,
            )
            scope_counts["session"] = len(session_records)
            recalled_records.extend(session_records)

        if user_id.strip():
            user_records = self._memory.recall(
                query=question,
                scope=MemoryScope.USER,
                top_k=self._settings.memory_top_k,
                user_id=user_id,
                session_id="",
            )
            scope_counts["user"] = len(user_records)
            recalled_records.extend(user_records)

        project_records = self._memory.recall(
            query=question,
            scope=MemoryScope.PROJECT,
            top_k=self._settings.memory_top_k,
            user_id="",
            session_id="",
        )
        scope_counts["project"] = len(project_records)
        recalled_records.extend(project_records)

        memory_context = self._format_memory_records(recalled_records)
        return memory_context, {
            "enabled": True,
            "recalled_count": len(recalled_records),
            "scope_counts": scope_counts,
            "preview": self._truncate(memory_context, max_len=220) if memory_context else "",
        }

    def _extract_user_profile_payload(self, *, question: str, answer: str) -> dict[str, Any] | None:
        # 这个函数用于通过LLM自动总结用户偏好并输出结构化画像。
        if not self._settings.openai_api_key:
            return None

        llm = ChatOpenAI(
            model=self._settings.openai_model,
            temperature=0,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        )
        messages = [
            SystemMessage(
                content=(
                    "你是用户画像提取器。"
                    "请根据本轮问答提取用户偏好，输出严格JSON，不允许额外文本。"
                    "JSON格式: {"
                    "\"risk_preference\":\"...\","
                    "\"investment_horizon\":\"...\","
                    "\"output_style\":\"...\","
                    "\"focus_topics\":[\"...\"],"
                    "\"language_preference\":\"...\","
                    "\"confidence\":0.0"
                    "}。"
                )
            ),
            HumanMessage(content=f"用户问题:\n{question}\n\n助手回答:\n{answer}"),
        ]
        result = llm.invoke(messages)
        payload = self._extract_json(str(result.content))
        return self._normalize_profile_payload(payload)

    @staticmethod
    def _normalize_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
        # 这个函数用于规整用户画像字段，防止脏值污染长期记忆。
        def clean_text(value: Any) -> str:
            text = str(value or "").strip()
            if text in {"无", "未知", "不确定", "N/A", "na"}:
                return ""
            return text

        confidence = float(payload.get("confidence", 0.0))
        topics_raw = payload.get("focus_topics", [])
        topics: list[str] = []
        if isinstance(topics_raw, list):
            for item in topics_raw:
                topic = clean_text(item)
                if topic and topic not in topics:
                    topics.append(topic)

        return {
            "risk_preference": clean_text(payload.get("risk_preference", "")),
            "investment_horizon": clean_text(payload.get("investment_horizon", "")),
            "output_style": clean_text(payload.get("output_style", "")),
            "focus_topics": topics[:8],
            "language_preference": clean_text(payload.get("language_preference", "")),
            "confidence": max(0.0, min(1.0, confidence)),
        }

    @staticmethod
    def _merge_profile_payload(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        # 这个函数用于把候选画像与已确认画像合并，避免低质量覆盖。
        merged = dict(existing)
        for key in ["risk_preference", "investment_horizon", "output_style", "language_preference"]:
            candidate_value = str(candidate.get(key, "")).strip()
            if candidate_value:
                merged[key] = candidate_value
            else:
                merged[key] = str(existing.get(key, "")).strip()

        topic_pool: list[str] = []
        for source in [existing.get("focus_topics", []), candidate.get("focus_topics", [])]:
            if isinstance(source, list):
                for topic in source:
                    normalized = str(topic).strip()
                    if normalized and normalized not in topic_pool:
                        topic_pool.append(normalized)
        merged["focus_topics"] = topic_pool[:8]
        merged["confidence"] = max(
            float(existing.get("confidence", 0.0)),
            float(candidate.get("confidence", 0.0)),
        )
        return merged

    @staticmethod
    def _profile_payload_to_text(payload: dict[str, Any]) -> str:
        # 这个函数用于把结构化画像格式化为可检索文本。
        return (
            f"风险偏好={payload.get('risk_preference', '')}; "
            f"投资期限={payload.get('investment_horizon', '')}; "
            f"输出风格={payload.get('output_style', '')}; "
            f"语言偏好={payload.get('language_preference', '')}; "
            f"关注主题={','.join(payload.get('focus_topics', []))}"
        )

    def _write_turn_memory(
        self,
        *,
        question: str,
        answer: str,
        route: RouteType,
        sources: list[str],
        user_id: str,
        session_id: str,
    ) -> str:
        # 这个函数用于写回会话、用户与项目记忆。
        if self._memory is None:
            return "disabled"

        status_fragments: list[str] = []
        try:
            expire_at = self._memory.default_expiration()
            self._memory.prune_expired(now=datetime.now(expire_at.tzinfo))

            if session_id.strip():
                self._memory.put(
                    scope=MemoryScope.SESSION,
                    text=f"Q: {question}\nA: {self._truncate(answer, max_len=260)}",
                    user_id=user_id,
                    session_id=session_id,
                    tags=["turn", route.value],
                    payload={"sources": sources[:3]},
                    importance=0.6,
                    expires_at=expire_at,
                )
                status_fragments.append("session:ok")
            else:
                status_fragments.append("session:skip")

            if user_id.strip():
                profile_payload = self._extract_user_profile_payload(question=question, answer=answer)
                if profile_payload is not None:
                    confidence = float(profile_payload.get("confidence", 0.0))
                    if confidence >= self._settings.memory_profile_min_confidence:
                        latest_rows = self._memory.list_recent(
                            scope=MemoryScope.USER,
                            limit=1,
                            user_id=user_id,
                            session_id="",
                        )
                        existing_payload = latest_rows[0].payload if latest_rows else {}
                        merged_payload = self._merge_profile_payload(existing_payload, profile_payload)
                        profile_text = self._profile_payload_to_text(merged_payload)
                        if latest_rows and latest_rows[0].text == profile_text:
                            status_fragments.append("user:unchanged")
                        else:
                            self._memory.put(
                                scope=MemoryScope.USER,
                                text=profile_text,
                                user_id=user_id,
                                session_id="",
                                tags=["user_profile", "auto_summary", "merged_update"],
                                payload=merged_payload,
                                importance=0.8,
                                expires_at=expire_at,
                            )
                            status_fragments.append("user:confirmed")
                    elif confidence >= self._profile_candidate_min_confidence:
                        candidate_text = self._profile_payload_to_text(profile_payload)
                        self._memory.put(
                            scope=MemoryScope.USER,
                            text=f"候选画像: {candidate_text}",
                            user_id=user_id,
                            session_id="",
                            tags=["user_profile_candidate", "auto_summary"],
                            payload=profile_payload,
                            importance=0.35,
                            expires_at=expire_at,
                        )
                        status_fragments.append("user:candidate")
                    else:
                        status_fragments.append("user:low_confidence")
                else:
                    status_fragments.append("user:no_payload")
            else:
                status_fragments.append("user:skip")

            if route in {RouteType.RAG_ONLY, RouteType.RAG_MCP}:
                project_summary = f"问题: {question}\n结论: {self._truncate(answer, max_len=220)}"
                self._memory.put(
                    scope=MemoryScope.PROJECT,
                    text=project_summary,
                    user_id="",
                    session_id="",
                    tags=["project_fact", route.value],
                    payload={"sources": sources[:5]},
                    importance=0.65,
                    expires_at=expire_at,
                )
                status_fragments.append("project:ok")
            else:
                status_fragments.append("project:skip")
        except Exception as exc:  # noqa: BLE001
            return f"failed:{exc}"

        return "ok|" + ",".join(status_fragments)

    def _finalize_with_memory(
        self,
        *,
        question: str,
        output: RouteOutput,
        user_id: str,
        session_id: str,
        memory_stats: dict[str, Any],
    ) -> RouteOutput:
        # 这个函数用于附加memory trace并执行写回。
        write_status = self._write_turn_memory(
            question=question,
            answer=output.answer,
            route=output.route,
            sources=output.sources,
            user_id=user_id,
            session_id=session_id,
        )
        data = dict(output.data)
        data["memory"] = {
            **memory_stats,
            "write_status": write_status,
            "embedding_error": self._memory.embedding_error if self._memory is not None else "",
        }
        return RouteOutput(
            output.route,
            output.answer,
            output.trace,
            output.sources,
            data,
        )

    def _synthesize_answer(
        self,
        *,
        question: str,
        docs: list[Document],
        mcp_records: list[dict[str, Any]],
        conversation_messages: list[BaseMessage] | None = None,
    ) -> str:
        # 这个函数用于在ReAct未直接给出final_answer时综合生成最终答案。
        if not self._settings.openai_api_key:
            if mcp_records:
                latest = mcp_records[-1].get("response", {})
                if latest.get("ok"):
                    result = latest.get("result", {})
                    return (
                        "依据当前可用信息，已完成测算。\n"
                        f"{result.get('formula', '无公式')}\n"
                        f"{result.get('interpretation', '无解读')}"
                    )
            return "依据当前资料无法确定。"

        llm = ChatOpenAI(
            model=self._settings.openai_model,
            temperature=0,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        )
        context = format_context(docs) if docs else "无检索片段"
        mcp_text = json.dumps(mcp_records, ensure_ascii=False, indent=2) if mcp_records else "无MCP结果"
        messages: list[BaseMessage] = [SystemMessage(content=AGENT_REACT_SYNTHESIS_PROMPT)]
        messages.extend(conversation_messages or [])
        messages.append(
            HumanMessage(
                content=(
                    f"用户问题:\n{question}\n\n"
                    f"RAG片段:\n{context}\n\n"
                    f"MCP结果:\n{mcp_text}\n\n"
                    "请输出最终回答。"
                )
            )
        )
        result = llm.invoke(messages)
        return str(result.content)

    def _handle_agent_level_react(
        self,
        question: str,
        intent: IntentResult,
        *,
        conversation_messages: list[BaseMessage] | None = None,
    ) -> RouteOutput:
        # 这个方法用于在Agent层统一执行Thought->Action->Observation循环。
        if not self._settings.openai_api_key:
            return self._dispatch_legacy(question, intent, conversation_messages=conversation_messages)

        llm = ChatOpenAI(
            model=self._settings.openai_model,
            temperature=0,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        )
        retriever = None

        rag_pool: list[Document] = []
        mcp_records: list[dict[str, Any]] = []
        trace: list[str] = [
            intent.reason,
            "已启用Agent级ReAct编排（Thought -> Action -> Observation）。",
        ]
        react_steps: list[dict[str, str]] = []
        observation_history: list[str] = ["尚无观察。"]
        answer: str | None = None

        for step in range(1, self._settings.react_max_steps + 1):
            planner_messages: list[BaseMessage] = [SystemMessage(content=AGENT_REACT_PLANNER_PROMPT)]
            planner_messages.extend(conversation_messages or [])
            planner_messages.append(
                HumanMessage(
                    content=(
                        f"用户问题:\n{question}\n\n"
                        f"当前路由提示:\n{intent.route.value}\n\n"
                        f"历史观察:\n{chr(10).join(observation_history)}\n\n"
                        "请输出本轮JSON决策。"
                    )
                )
            )
            planner_result = llm.invoke(planner_messages)
            planner_text = str(planner_result.content)
            plan = self._extract_json(planner_text)

            thought = str(plan.get("thought", "")).strip() or "基于当前观察继续推进。"
            action = str(plan.get("action", "")).strip().lower()
            action_input = str(plan.get("action_input", "")).strip()
            final_answer = str(plan.get("final_answer", "")).strip()

            trace.append(f"ReAct: Step {step} | Thought: {thought}")

            if action == "finish":
                answer = final_answer or self._synthesize_answer(
                    question=question,
                    docs=rag_pool,
                    mcp_records=mcp_records,
                    conversation_messages=conversation_messages,
                )
                observation = "结束循环并产出最终回答。"
                trace.append(f"ReAct: Step {step} | Action: finish")
                trace.append(f"ReAct: Step {step} | Observation: {observation}")
                react_steps.append(
                    {
                        "step": str(step),
                        "thought": thought,
                        "action": "finish",
                        "observation": observation,
                    }
                )
                break

            if action == "rag_retrieve":
                query = action_input or question
                if retriever is None:
                    # 这个分支用于仅在实际检索时加载Embedding与向量库。
                    retriever = get_retriever(self._settings)
                step_docs: list[Document] = retriever.invoke(query)
                self._append_unique_docs(rag_pool, step_docs)
                observation = self._build_rag_observation(step_docs)
                observation_history.append(
                    f"Step {step} rag_retrieve(query={query})\n{observation}"
                )
                trace.append(f"ReAct: Step {step} | Action: rag_retrieve | query={query}")
                trace.append(f"ReAct: Step {step} | Observation: {observation}")
                react_steps.append(
                    {
                        "step": str(step),
                        "thought": thought,
                        "action": f"rag_retrieve | query={query}",
                        "observation": observation,
                    }
                )
                continue

            if action == "mcp_call":
                instruction = action_input or question
                request = build_request_from_question(instruction, self._settings)
                response = self._mcp.handle(request)
                mcp_records.append({"request": request, "response": response})
                observation = self._build_mcp_observation(response)
                observation_history.append(
                    f"Step {step} mcp_call(request={json.dumps(request, ensure_ascii=False)})\n{observation}"
                )
                trace.append(
                    f"ReAct: Step {step} | Action: mcp_call | request={json.dumps(request, ensure_ascii=False)}"
                )
                trace.append(f"ReAct: Step {step} | Observation: {observation}")
                react_steps.append(
                    {
                        "step": str(step),
                        "thought": thought,
                        "action": f"mcp_call | request={json.dumps(request, ensure_ascii=False)}",
                        "observation": observation,
                    }
                )
                continue

            observation = f"未识别动作action={action}，跳过执行。"
            observation_history.append(f"Step {step} unknown_action\n{observation}")
            trace.append(f"ReAct: Step {step} | Action: unknown({action})")
            trace.append(f"ReAct: Step {step} | Observation: {observation}")
            react_steps.append(
                {
                    "step": str(step),
                    "thought": thought,
                    "action": f"unknown({action})",
                    "observation": observation,
                }
            )

        if answer is None:
            # 这个分支用于步数耗尽时自动合成答案，保证可用性。
            answer = self._synthesize_answer(
                question=question,
                docs=rag_pool,
                mcp_records=mcp_records,
                conversation_messages=conversation_messages,
            )
            trace.append("ReAct: 未在限制步数内finish，已自动合成最终答案。")

        return RouteOutput(
            intent.route,
            answer,
            trace,
            self._format_sources(rag_pool),
            {
                "react_steps": react_steps,
                "rag_docs_count": len(rag_pool),
                "mcp_records": mcp_records,
            },
        )

    def _handle_mcp_only(self, question: str, intent: IntentResult) -> RouteOutput:
        # 这个方法用于执行纯MCP计算路径。
        request = build_request_from_question(question, self._settings)
        response = self._mcp.handle(request)
        if not response["ok"]:
            answer = f"MCP计算失败：{response['error']['message']}"
            return RouteOutput(intent.route, answer, [intent.reason], [], {"mcp": response})

        result = response["result"]
        answer = (
            "### 计算结果\n"
            f"{result['formula']}\n"
            "### 简要解读\n"
            f"{result['interpretation']}"
        )
        return RouteOutput(intent.route, answer, [intent.reason], [], {"mcp": response})

    def _handle_rag_only(
        self,
        question: str,
        intent: IntentResult,
        *,
        conversation_messages: list[BaseMessage] | None = None,
    ) -> RouteOutput:
        # 这个方法用于执行纯RAG检索问答路径。
        rag_result = ask_with_rag(self._settings, question, conversation_history=conversation_messages)
        trace = [intent.reason]
        react_trace = rag_result.get("react_trace", [])
        if isinstance(react_trace, list):
            trace.extend([f"ReAct: {item}" for item in react_trace])
        return RouteOutput(
            intent.route,
            rag_result["answer"],
            trace,
            rag_result["sources"],
            {"rag": rag_result},
        )

    def _handle_rag_mcp(
        self,
        question: str,
        intent: IntentResult,
        *,
        conversation_messages: list[BaseMessage] | None = None,
    ) -> RouteOutput:
        # 这个方法用于执行先RAG后MCP的联合推理路径。
        rag_result = ask_with_rag(self._settings, question, conversation_history=conversation_messages)
        mcp_request = build_request_from_question(question, self._settings)
        mcp_response = self._mcp.handle(mcp_request)

        if not mcp_response["ok"]:
            answer = (
                "已完成行业资料检索，但量化计算失败。\n"
                f"失败原因：{mcp_response['error']['message']}"
            )
            return RouteOutput(
                intent.route,
                answer,
                [intent.reason, "RAG完成，MCP失败。"],
                rag_result["sources"],
                {"rag": rag_result, "mcp": mcp_response},
            )

        benchmark_info = self._extract_benchmark_from_rag(
            question=question,
            tool_name=mcp_request["tool_name"],
            rag_answer=str(rag_result.get("answer", "")),
        )
        benchmark = benchmark_info["value"]
        calc_value = float(mcp_response["result"]["value"])
        compare_text = self._build_compare_text(mcp_request["tool_name"], calc_value, benchmark)

        answer = (
            "### 一、量化测算结果（MCP）\n"
            f"{mcp_response['result']['formula']}\n"
            "### 二、行业/政策要点（RAG）\n"
            f"{rag_result['answer']}\n"
            "### 三、对标结论（联合推理）\n"
            f"{compare_text}"
        )

        trace = [intent.reason, "先执行RAG提取基准，再执行MCP量化并自动对比。"]
        react_trace = rag_result.get("react_trace", [])
        if isinstance(react_trace, list):
            trace.extend([f"ReAct: {item}" for item in react_trace])
        if benchmark_info["source"] == "rag":
            trace.append(f"行业基准抽取来源：RAG文本（{benchmark_info['evidence']}）。")
        elif benchmark_info["source"] == "question":
            trace.append("行业基准未在RAG文本中稳定抽取，已回退为问题显式数字。")
        else:
            trace.append("未抽取到行业基准，仅输出测算值与建议。")

        return RouteOutput(
            intent.route,
            answer,
            trace,
            rag_result["sources"],
            {
                "rag": rag_result,
                "mcp": mcp_response,
                "benchmark": benchmark,
                "benchmark_source": benchmark_info["source"],
                "benchmark_evidence": benchmark_info["evidence"],
            },
        )

    def _handle_general(
        self,
        question: str,
        intent: IntentResult,
        *,
        conversation_messages: list[BaseMessage] | None = None,
    ) -> RouteOutput:
        # 这个方法用于回复无需检索与计算的基础常识问题。
        answer = self._general_answer(question, conversation_messages=conversation_messages)
        return RouteOutput(intent.route, answer, [intent.reason], [], {})

    @staticmethod
    def _extract_benchmark_number(question: str) -> float | None:
        # 这个函数用于从问题文本中提取行业基准数值。
        number_texts = re.findall(r"[0-9]+(?:\.[0-9]+)?", question)
        if len(number_texts) < 3:
            return None
        try:
            return float(number_texts[0])
        except ValueError:
            return None

    def _extract_benchmark_from_rag(
        self,
        *,
        question: str,
        tool_name: str,
        rag_answer: str,
    ) -> dict[str, Any]:
        # 这个函数用于优先从RAG答案文本抽取行业基准，失败再回退到问题数字。
        candidates: list[float] = []
        evidence = ""

        if tool_name == "calc_pe":
            pe_pattern = re.compile(
                r"(?:行业|同业|股份行)[^\n。；]*?(?:平均|基准|中枢)?[^\n。；]*?(?:PE|市盈率)\s*[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)",
                flags=re.IGNORECASE,
            )
            matches = pe_pattern.findall(rag_answer)
            candidates = [float(value) for value in matches]
            if matches:
                evidence = f"提取到PE候选={matches[0]}"

        elif tool_name == "calc_debt_ratio":
            debt_pattern = re.compile(
                r"(?:行业|同业|股份行)[^\n。；]*?(?:平均|基准|区间)?[^\n。；]*?资产负债率\s*[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)\s*%?",
                flags=re.IGNORECASE,
            )
            matches = debt_pattern.findall(rag_answer)
            candidates = [float(value) for value in matches]
            if matches:
                evidence = f"提取到负债率候选={matches[0]}%"

        if candidates:
            return {
                "value": candidates[0],
                "source": "rag",
                "evidence": evidence,
            }

        fallback = self._extract_benchmark_number(question)
        if fallback is not None:
            return {
                "value": fallback,
                "source": "question",
                "evidence": "问题中的首个候选基准数字",
            }

        return {
            "value": None,
            "source": "none",
            "evidence": "未命中",
        }

    @staticmethod
    def _build_compare_text(tool_name: str, calc_value: float, benchmark: float | None) -> str:
        # 这个函数用于输出统一的对标分析描述。
        if tool_name == "calc_pe":
            if benchmark is None:
                return f"个股静态PE约为{calc_value:.2f}倍，建议结合检索到的行业估值中枢进一步判断。"
            gap = calc_value - benchmark
            trend = "高于" if gap > 0 else "低于" if gap < 0 else "等于"
            return (
                f"行业基准PE约{benchmark:.2f}倍，个股PE约{calc_value:.2f}倍，"
                f"当前个股{trend}行业中枢{abs(gap):.2f}倍。"
            )

        if tool_name == "calc_debt_ratio":
            ratio_percent = calc_value * 100
            if benchmark is None:
                return f"个股资产负债率约为{ratio_percent:.2f}%，建议结合行业区间与负债结构综合评估。"
            gap = ratio_percent - benchmark
            trend = "高于" if gap > 0 else "低于" if gap < 0 else "等于"
            return (
                f"行业基准资产负债率约{benchmark:.2f}%，个股约{ratio_percent:.2f}%，"
                f"当前个股{trend}行业均值{abs(gap):.2f}个百分点。"
            )

        return "已完成检索与测算，建议结合行业文本进一步判断。"

    def _general_answer(
        self,
        question: str,
        *,
        conversation_messages: list[BaseMessage] | None = None,
    ) -> str:
        # 这个函数用于提供轻量常识兜底回复。
        usable_api_key = self._settings.openai_api_key.strip() not in {
            "",
            "your_deepseek_api_key_here",
            "your_openai_api_key_here",
        }
        if conversation_messages and usable_api_key:
            llm = ChatOpenAI(
                model=self._settings.openai_model,
                temperature=0,
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url,
            )
            messages: list[BaseMessage] = [
                SystemMessage(content="你是金融投研助手。请结合当前聊天历史准确回答，不要编造对话内容。")
            ]
            messages.extend(conversation_messages)
            messages.append(HumanMessage(content=question))
            try:
                return str(llm.invoke(messages).content)
            except Exception:
                pass

        if "拨备覆盖率" in question:
            return (
                "拨备覆盖率是银行贷款损失准备与不良贷款余额的比值，"
                "通常用于衡量风险缓冲能力，数值越高一般代表抗风险能力越强。"
            )
        return "该问题更适合基础金融常识解释，请补充更具体的指标或场景以便深入分析。"

    def _update_chat_profile(self, *, question: str, answer: str, user_id: str) -> str:
        # 这个函数用于仅在用户明确表达偏好时更新跨会话画像。
        if not user_id.strip():
            return "skip:no_user"
        preference_markers = ["我偏好", "我喜欢", "以后请", "后续请", "请记住", "回答风格", "风险偏好", "投资期限"]
        if not any(marker in question for marker in preference_markers):
            return "skip:not_explicit"
        try:
            candidate = self._extract_user_profile_payload(question=question, answer=answer)
            if candidate is None:
                return "skip:no_payload"
            confidence = float(candidate.get("confidence", 0.0))
            if confidence < self._settings.memory_profile_min_confidence:
                return "skip:low_confidence"
            chat_memory = self._get_chat_memory()
            existing = chat_memory.get_profile(user_id=user_id)
            merged = self._merge_profile_payload(existing.preferences if existing else {}, candidate)
            chat_memory.upsert_profile(
                user_id=user_id,
                preferences=merged,
                confidence=confidence,
            )
            return "updated"
        except Exception as exc:  # noqa: BLE001
            return f"failed:{exc}"
