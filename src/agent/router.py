# 这个文件用于实现阶段四的统一执行路由与结果编排。
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.agent.intent import IntentResult, RouteType, classify_intent
from src.config import Settings
from src.mcp import MCPDispatcher
from src.mcp.nl_router import build_request_from_question
from src.rag.qa_chain import ask_with_rag


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

        if intent.route == RouteType.MCP_ONLY:
            return self._handle_mcp_only(question, intent)

        if intent.route == RouteType.RAG_ONLY:
            return self._handle_rag_only(question, intent)

        if intent.route == RouteType.RAG_MCP:
            return self._handle_rag_mcp(question, intent)

        return self._handle_general(question, intent)

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
        return RouteOutput(
            intent.route,
            rag_result["answer"],
            [intent.reason],
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
