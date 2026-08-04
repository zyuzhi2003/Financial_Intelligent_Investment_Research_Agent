# 这个脚本用于独立验证阶段四意图识别与路由分流逻辑。
from __future__ import annotations

from pathlib import Path
import sys

# 这个代码块用于把项目根目录加入导入路径，保证脚本可独立执行。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.intent import IntentResult, RouteType, classify_intent
from src.agent.router import AgentRouter
from src.config import Settings


def _assert_equal(actual, expected, message: str) -> None:
    # 这个函数用于统一断言输出。
    if actual != expected:
        raise AssertionError(f"{message}: actual={actual}, expected={expected}")


def main() -> int:
    # 这个函数用于执行阶段四路由判定测试。
    cases = [
        (
            "2024年银行业整体净息差走势如何？",
            RouteType.RAG_ONLY,
        ),
        (
            "中信银行现价8.25元，每股收益EPS0.68，计算静态PE",
            RouteType.MCP_ONLY,
        ),
        (
            "股份行行业平均PE13.5倍，中信银行股价8.25元，EPS0.68，判断估值高低",
            RouteType.RAG_MCP,
        ),
        (
            "什么是银行拨备覆盖率？",
            RouteType.GENERAL,
        ),
    ]

    for question, expected_route in cases:
        result = classify_intent(question)
        _assert_equal(result.route, expected_route, f"路由识别失败: {question}")

    project_root = Path(__file__).resolve().parent.parent
    settings = Settings(
        knowledge_dir=project_root / "Knowledge",
        persist_dir=project_root / "data" / "chroma",
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        hf_endpoint="https://hf-mirror.com",
        enable_ocr=True,
        ocr_dpi=200,
        min_page_chars=20,
        chunk_size=800,
        chunk_overlap=120,
        top_k=4,
        collection_name="bank_research_reports",
        openai_base_url="https://api.deepseek.com/v1",
        openai_model="deepseek-chat",
        openai_api_key="",
    )
    router = AgentRouter(settings)

    stabilized = router._stabilize_intent(
        "请结合研报，股份行行业平均PE13.5倍，中信银行股价8.25元，EPS0.68，判断估值高低",
        IntentResult(route=RouteType.MCP_ONLY, reason="mock"),
    )
    _assert_equal(stabilized.route, RouteType.RAG_MCP, "路由稳定化应把研报+测算纠偏到RAG_MCP")

    stabilized_mcp = router._stabilize_intent(
        "中信银行现价8.25元，每股收益EPS0.68，计算静态PE",
        IntentResult(route=RouteType.RAG_MCP, reason="mock"),
    )
    _assert_equal(stabilized_mcp.route, RouteType.MCP_ONLY, "路由稳定化应把纯测算纠偏到MCP_ONLY")

    pe_benchmark = router._extract_benchmark_from_rag(
        question="判断估值高低",
        tool_name="calc_pe",
        rag_answer="股份行行业平均PE为13.5倍，估值中枢相对平稳。",
    )
    _assert_equal(pe_benchmark["source"], "rag", "PE基准应优先来自RAG文本")
    _assert_equal(pe_benchmark["value"], 13.5, "PE基准提取值异常")

    debt_benchmark = router._extract_benchmark_from_rag(
        question="结合2024股份行负债标准分析",
        tool_name="calc_debt_ratio",
        rag_answer="2024年股份行平均资产负债率约91.2%，行业合理区间89%-93%。",
    )
    _assert_equal(debt_benchmark["source"], "rag", "负债率基准应优先来自RAG文本")
    _assert_equal(debt_benchmark["value"], 91.2, "负债率基准提取值异常")

    print("Agent路由测试通过：四类意图识别均符合预期。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
