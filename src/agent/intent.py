# 这个文件用于实现阶段四的四分类意图识别。
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from src.config import Settings


class RouteType(str, Enum):
    # 这个枚举用于标识四类执行路径。
    RAG_ONLY = "rag_only"
    MCP_ONLY = "mcp_only"
    RAG_MCP = "rag_mcp"
    GENERAL = "general"


@dataclass(frozen=True)
class IntentResult:
    # 这个字段用于返回最终路由类型。
    route: RouteType
    # 这个字段用于解释命中的规则，方便调试。
    reason: str


def _rule_based_classify(question: str) -> IntentResult:
    # 这个函数用于提供稳定可解释的规则分流兜底。
    text = question.strip()
    lower = text.lower()

    has_calc_word = any(keyword in lower for keyword in ["计算", "测算", "算", "求", "估算"])
    has_pe_word = any(keyword in lower for keyword in ["pe", "市盈率"])
    has_debt_word = "资产负债率" in text
    has_numeric_data = any(char.isdigit() for char in text)

    has_industry_word = any(
        keyword in text
        for keyword in ["行业", "平均", "中枢", "基准", "同业", "区间", "股份行"]
    )
    has_research_word = any(
        keyword in text
        for keyword in ["研报", "政策", "净息差", "不良率", "资本充足率", "财报", "AUM", "银行业"]
    )
    has_financial_fact = any(
        keyword in text for keyword in ["收入", "营收", "净利润", "业绩", "贷款", "资产", "负债"]
    )
    has_time_marker = bool(re.search(r"20\d{2}|上半年|下半年|一季度|二季度|三季度|四季度|[0-9]{2}年", text))
    has_company_or_industry = any(keyword in text for keyword in ["银行", "公司", "股份行", "上市"])

    if (has_pe_word or has_debt_word) and has_numeric_data and has_industry_word:
        return IntentResult(
            route=RouteType.RAG_MCP,
            reason="检测到计算指标+数值输入+行业对标词，走RAG+MCP联合路径。",
        )

    if (has_pe_word or has_debt_word) and has_numeric_data and (has_calc_word or "计算" in text):
        return IntentResult(
            route=RouteType.MCP_ONLY,
            reason="检测到财务指标计算意图与数值输入，走MCP路径。",
        )

    if has_research_word:
        return IntentResult(
            route=RouteType.RAG_ONLY,
            reason="检测到研报/政策/基本面检索词，走RAG路径。",
        )

    if has_company_or_industry and (has_financial_fact or has_time_marker):
        return IntentResult(
            route=RouteType.RAG_ONLY,
            reason="检测到银行/公司财务事实查询，走RAG路径。",
        )

    return IntentResult(
        route=RouteType.GENERAL,
        reason="未命中检索或计算强特征，走常识回复路径。",
    )


def _coerce_route(route_name: str) -> RouteType | None:
    # 这个函数用于把模型输出的字符串安全映射为路由枚举。
    normalized = route_name.strip().lower()
    route_map = {
        "rag_only": RouteType.RAG_ONLY,
        "mcp_only": RouteType.MCP_ONLY,
        "rag_mcp": RouteType.RAG_MCP,
        "general": RouteType.GENERAL,
    }
    return route_map.get(normalized)


def _llm_classify(question: str, settings: Settings) -> IntentResult | None:
    # 这个函数用于通过LLM执行四分类意图识别。
    if not settings.openai_api_key:
        return None

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    messages = [
        SystemMessage(
            content=(
                "你是金融Agent的路由分类器。"
                "请在 rag_only/mcp_only/rag_mcp/general 中选择一个最合适的route，"
                "并输出严格JSON，不要输出额外文本。"
                "JSON格式: {\"route\":\"...\",\"reason\":\"...\"}。"
                "规则要点："
                "1) 仅资料查询/政策解读/基本面分析 => rag_only；"
                "2) 仅数值计算（PE/资产负债率）=> mcp_only；"
                "3) 同时要求行业对标+数值测算 => rag_mcp；"
                "4) 常识问答 => general。"
            )
        ),
        HumanMessage(content=question),
    ]

    result = llm.invoke(messages)
    content = result.content if isinstance(result.content, str) else str(result.content)
    cleaned = content.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        return None

    route_name = str(payload.get("route", "")).strip()
    route = _coerce_route(route_name)
    if route is None:
        return None

    reason = str(payload.get("reason", "")).strip() or "LLM完成意图识别。"
    return IntentResult(route=route, reason=f"LLM识别：{reason}")


def classify_intent(question: str, settings: Settings | None = None) -> IntentResult:
    # 这个函数用于优先使用LLM分类，并在失败时自动回退规则分流。
    rule_result = _rule_based_classify(question)
    if settings is None:
        return rule_result

    try:
        llm_result = _llm_classify(question, settings)
        if llm_result is not None:
            return llm_result
    except Exception:  # noqa: BLE001
        pass

    return IntentResult(
        route=rule_result.route,
        reason=f"规则兜底：{rule_result.reason}",
    )
