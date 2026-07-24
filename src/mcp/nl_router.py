# 这个文件用于把中文自然语言问题转换为MCP标准JSON请求。
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import Settings


def _extract_number_by_label(text: str, labels: list[str]) -> float | None:
    # 这个函数用于按字段标签优先提取对应数值。
    for label in labels:
        pattern = rf"{label}\s*[=:：]?\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _rule_based_request(question: str) -> dict:
    # 这个函数用于提供规则解析兜底。
    normalized = question.strip()
    if not normalized:
        raise ValueError("问题不能为空。")

    has_pe_keyword = any(keyword in normalized.lower() for keyword in ["pe", "市盈率"])
    has_valuation_keyword = any(keyword in normalized for keyword in ["估值", "估值高低", "高估", "低估", "估值水平"])
    has_price_hint = any(keyword in normalized for keyword in ["现价", "股价", "价格"])
    has_eps_hint = any(keyword in normalized.lower() for keyword in ["eps", "每股收益"])
    is_pe_question = has_pe_keyword or (has_valuation_keyword and (has_price_hint or has_eps_hint))
    is_debt_question = "资产负债率" in normalized

    all_numbers = [float(number) for number in re.findall(r"[0-9]+(?:\.[0-9]+)?", normalized)]

    if is_pe_question:
        price = _extract_number_by_label(normalized, ["现价", "股价", "价格"])
        eps = _extract_number_by_label(normalized, ["eps", "每股收益"])
        if price is None or eps is None:
            if len(all_numbers) >= 2:
                price = all_numbers[0]
                eps = all_numbers[1]
            else:
                raise ValueError("PE问题缺少数值，请至少提供股价和EPS。")

        return {
            "tool_name": "calc_pe",
            "params": {
                "price": price,
                "eps": eps,
            },
        }

    if is_debt_question:
        total_asset = _extract_number_by_label(normalized, ["总资产"])
        total_liability = _extract_number_by_label(normalized, ["总负债"])
        if total_asset is None or total_liability is None:
            if len(all_numbers) >= 2:
                total_asset = all_numbers[0]
                total_liability = all_numbers[1]
            else:
                raise ValueError("资产负债率问题缺少数值，请至少提供总资产和总负债。")

        return {
            "tool_name": "calc_debt_ratio",
            "params": {
                "total_asset": total_asset,
                "total_liability": total_liability,
            },
        }

    raise ValueError("暂不支持该问题类型，请使用PE或资产负债率相关问题。")


def _coerce_float(name: str, value: object) -> float:
    # 这个函数用于把LLM返回参数统一转换为浮点数。
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"字段{name}必须是数值。") from exc


def _normalize_request(payload: dict) -> dict:
    # 这个函数用于校验和规整LLM生成的请求结构。
    tool_name = str(payload.get("tool_name", "")).strip().lower()
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("字段params必须是对象。")

    if tool_name == "calc_pe":
        return {
            "tool_name": "calc_pe",
            "params": {
                "price": _coerce_float("price", params.get("price")),
                "eps": _coerce_float("eps", params.get("eps")),
            },
        }

    if tool_name == "calc_debt_ratio":
        return {
            "tool_name": "calc_debt_ratio",
            "params": {
                "total_asset": _coerce_float("total_asset", params.get("total_asset")),
                "total_liability": _coerce_float("total_liability", params.get("total_liability")),
            },
        }

    raise ValueError("LLM未返回受支持的tool_name。")


def _llm_request(question: str, settings: Settings) -> dict:
    # 这个函数用于优先通过LLM生成MCP工具请求。
    if not settings.openai_api_key:
        raise ValueError("缺少模型API Key，无法执行LLM解析。")

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    messages = [
        SystemMessage(
            content=(
                "你是金融MCP工具解析器。"
                "请从用户问题中选择一个工具并抽取参数，"
                "仅输出严格JSON，不允许额外文本。"
                "工具仅支持calc_pe和calc_debt_ratio。"
                "- calc_pe参数: price, eps；"
                "- calc_debt_ratio参数: total_asset, total_liability。"
                "若问题不属于这两类工具，输出"
                "{\"tool_name\":\"unsupported\",\"params\":{},\"reason\":\"...\"}。"
                "若问题表达估值高低且给出股价/EPS，默认使用calc_pe。"
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
        raise ValueError("LLM解析结果不是JSON对象。")

    if str(payload.get("tool_name", "")).strip().lower() == "unsupported":
        reason = str(payload.get("reason", "")).strip() or "问题不属于已支持工具。"
        raise ValueError(f"暂不支持该问题类型：{reason}")

    return _normalize_request(payload)


def build_request_from_question(question: str, settings: Settings | None = None) -> dict:
    # 这个函数用于LLM-first地解析请求，失败时自动回退规则解析。
    if settings is not None:
        try:
            return _llm_request(question, settings)
        except Exception:  # noqa: BLE001
            pass
    return _rule_based_request(question)
