# 这个脚本用于独立验证阶段四意图识别与路由分流逻辑。
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

from langchain_core.messages import BaseMessage
# 这个代码块用于把项目根目录加入导入路径，保证脚本可独立执行。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.intent import IntentResult, RouteType, classify_intent
from src.agent.router import AgentRouter, RouteOutput
from src.config import Settings


def _assert_equal(actual, expected, message: str) -> None:
    # 这个函数用于统一断言输出。
    if actual != expected:
        raise AssertionError(f"{message}: actual={actual}, expected={expected}")


class CapturingRouter(AgentRouter):
    # 这个测试替身用于捕获进入回答链路的标准聊天消息，避免访问真实模型。
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.captured_histories: list[list[BaseMessage]] = []

    def _handle_agent_level_react(
        self,
        question: str,
        intent: IntentResult,
        *,
        conversation_messages: list[BaseMessage] | None = None,
    ) -> RouteOutput:
        self.captured_histories.append(list(conversation_messages or []))
        return RouteOutput(intent.route, f"回答:{question}", [intent.reason], [], {})


class ProfileRouter(CapturingRouter):
    # 这个测试替身用于稳定返回高置信偏好画像。
    def _extract_user_profile_payload(self, *, question: str, answer: str) -> dict:
        _ = (question, answer)
        return {
            "risk_preference": "稳健",
            "investment_horizon": "中长期",
            "output_style": "要点式",
            "focus_topics": ["银行"],
            "language_preference": "中文",
            "confidence": 0.95,
        }


class FailingProfileRouter(CapturingRouter):
    # 这个测试替身用于验证画像提取失败不会破坏聊天消息。
    def _extract_user_profile_payload(self, *, question: str, answer: str) -> dict:
        _ = (question, answer)
        raise TimeoutError("mock profile timeout")


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

    # 这个代码块用于验证chat传递标准消息历史，而ask保持无状态。
    temp_dir = Path(tempfile.mkdtemp(prefix="router-chat-test-"))
    try:
        chat_settings = Settings(
            knowledge_dir=project_root / "Knowledge",
            persist_dir=project_root / "data" / "chroma",
            embedding_model_name="unused",
            hf_endpoint="",
            enable_ocr=False,
            ocr_dpi=200,
            min_page_chars=20,
            chunk_size=800,
            chunk_overlap=120,
            top_k=4,
            collection_name="unused",
            openai_base_url="",
            openai_model="unused",
            openai_api_key="",
            enable_react=True,
            memory_dir=temp_dir,
        )
        capturing_router = CapturingRouter(chat_settings)
        capturing_router.chat("第一个问题", user_id="u1", session_id="s1")
        capturing_router.chat("我上一个问题是什么", user_id="u1", session_id="s1")
        second_history = capturing_router.captured_histories[1]
        _assert_equal(
            [(message.type, str(message.content)) for message in second_history],
            [("human", "第一个问题"), ("ai", "回答:第一个问题")],
            "chat应按原始顺序传入上一轮标准消息",
        )

        capturing_router.ask("无状态问题", user_id="u1", session_id="ask-session")
        _assert_equal(
            capturing_router._get_chat_memory().list_messages(session_id="ask-session", limit=10),
            [],
            "ask不应读写chat记忆",
        )
        capturing_router._get_chat_memory().close()

        profile_router = ProfileRouter(chat_settings)
        profile_output = profile_router.chat("以后请用要点式回答", user_id="u-profile", session_id="profile-s1")
        _assert_equal(profile_output.data["chat_memory"]["profile_write_status"], "updated", "明确偏好应更新画像")
        saved_profile = profile_router._get_chat_memory().get_profile(user_id="u-profile")
        _assert_equal(saved_profile.preferences["output_style"] if saved_profile else "", "要点式", "画像未跨session保存")
        profile_router._get_chat_memory().close()

        failing_router = FailingProfileRouter(chat_settings)
        failed_profile_output = failing_router.chat("以后请简洁回答", user_id="u-fail", session_id="fail-s1")
        _assert_equal(
            str(failed_profile_output.data["chat_memory"]["profile_write_status"]).startswith("failed:"),
            True,
            "画像异常应记录失败状态",
        )
        failed_messages = failing_router._get_chat_memory().list_messages(session_id="fail-s1", limit=10)
        _assert_equal([message.role for message in failed_messages], ["user", "assistant"], "画像异常不应影响消息落库")
        failing_router._get_chat_memory().close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Agent路由测试通过：意图识别与标准聊天消息历史均符合预期。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
