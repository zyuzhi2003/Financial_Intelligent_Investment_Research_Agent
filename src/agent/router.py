# 这个文件用于实现阶段四的统一执行路由与结果编排。
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.intent import IntentResult, RouteType, classify_intent
from src.config import Settings
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

    def route_only(self, question: str) -> IntentResult:
        # 这个方法用于仅返回意图识别结果。
        return classify_intent(question, self._settings)

    def ask(self, question: str) -> RouteOutput:
        # 这个方法用于执行四类场景自动分流。
        intent = classify_intent(question, self._settings)

        if self._settings.enable_react:
            # 这个分支用于优先启用Agent级ReAct编排；失败时回退原有路径。
            try:
                return self._handle_agent_level_react(question, intent)
            except Exception:  # noqa: BLE001
                pass

        return self._dispatch_legacy(question, intent)

    def _dispatch_legacy(self, question: str, intent: IntentResult) -> RouteOutput:
        # 这个方法用于保留原有稳定路由逻辑作为兜底。
        if intent.route == RouteType.MCP_ONLY:
            return self._handle_mcp_only(question, intent)

        if intent.route == RouteType.RAG_ONLY:
            return self._handle_rag_only(question, intent)

        if intent.route == RouteType.RAG_MCP:
            return self._handle_rag_mcp(question, intent)

        return self._handle_general(question, intent)

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

    def _synthesize_answer(
        self,
        *,
        question: str,
        docs: list[Document],
        mcp_records: list[dict[str, Any]],
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
        messages = [
            SystemMessage(content=AGENT_REACT_SYNTHESIS_PROMPT),
            HumanMessage(
                content=(
                    f"用户问题:\n{question}\n\n"
                    f"RAG片段:\n{context}\n\n"
                    f"MCP结果:\n{mcp_text}\n\n"
                    "请输出最终回答。"
                )
            ),
        ]
        result = llm.invoke(messages)
        return str(result.content)

    def _handle_agent_level_react(self, question: str, intent: IntentResult) -> RouteOutput:
        # 这个方法用于在Agent层统一执行Thought->Action->Observation循环。
        if not self._settings.openai_api_key:
            return self._dispatch_legacy(question, intent)

        llm = ChatOpenAI(
            model=self._settings.openai_model,
            temperature=0,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        )
        retriever = get_retriever(self._settings)

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
            planner_messages = [
                SystemMessage(content=AGENT_REACT_PLANNER_PROMPT),
                HumanMessage(
                    content=(
                        f"用户问题:\n{question}\n\n"
                        f"当前路由提示:\n{intent.route.value}\n\n"
                        f"历史观察:\n{chr(10).join(observation_history)}\n\n"
                        "请输出本轮JSON决策。"
                    )
                ),
            ]
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

    def _handle_rag_only(self, question: str, intent: IntentResult) -> RouteOutput:
        # 这个方法用于执行纯RAG检索问答路径。
        rag_result = ask_with_rag(self._settings, question)
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

    def _handle_rag_mcp(self, question: str, intent: IntentResult) -> RouteOutput:
        # 这个方法用于执行先RAG后MCP的联合推理路径。
        rag_result = ask_with_rag(self._settings, question)
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

    def _handle_general(self, question: str, intent: IntentResult) -> RouteOutput:
        # 这个方法用于回复无需检索与计算的基础常识问题。
        answer = self._general_answer(question)
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

    @staticmethod
    def _general_answer(question: str) -> str:
        # 这个函数用于提供轻量常识兜底回复。
        if "拨备覆盖率" in question:
            return (
                "拨备覆盖率是银行贷款损失准备与不良贷款余额的比值，"
                "通常用于衡量风险缓冲能力，数值越高一般代表抗风险能力越强。"
            )
        return "该问题更适合基础金融常识解释，请补充更具体的指标或场景以便深入分析。"
