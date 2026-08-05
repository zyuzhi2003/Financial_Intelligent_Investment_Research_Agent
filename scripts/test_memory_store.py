# 这个脚本用于独立验证记忆模块的基础存储能力。
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import shutil
import sys
import tempfile

# 这个代码块用于把项目根目录加入导入路径，保证脚本可独立执行。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings
from src.memory import ChatMemoryStore, HybridMemoryStore, MemoryScope


def _assert_true(condition: bool, message: str) -> None:
    # 这个函数用于统一断言输出。
    if not condition:
        raise AssertionError(message)


def _build_settings(memory_dir: Path) -> Settings:
    # 这个函数用于构造测试专用Settings。
    return Settings(
        knowledge_dir=PROJECT_ROOT / "Knowledge",
        persist_dir=PROJECT_ROOT / "data" / "chroma",
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
        memory_enabled=True,
        memory_backend="sqlite",
        memory_dir=memory_dir,
        memory_top_k=3,
        memory_ttl_days=30,
        memory_collection="agent_memory_test",
        memory_profile_min_confidence=0.7,
    )


def main() -> int:
    # 这个函数用于执行记忆模块回归测试。
    temp_dir = Path(tempfile.mkdtemp(prefix="memory-test-"))
    store = None
    try:
        settings = _build_settings(temp_dir)
        store = HybridMemoryStore(settings)

        # 这个代码块用于验证聊天消息无需向量模型即可按会话顺序持久化。
        chat_store = ChatMemoryStore(temp_dir)
        chat_store.ensure_session(session_id="chat-1", user_id="u1")
        chat_store.append_message(session_id="chat-1", user_id="u1", role="user", content="第一个问题")
        chat_store.append_message(session_id="chat-1", user_id="u1", role="assistant", content="第一个回答")
        chat_store.append_message(session_id="chat-1", user_id="u1", role="user", content="第二个问题")
        messages = chat_store.list_messages(session_id="chat-1", limit=10)
        _assert_true([item.content for item in messages] == ["第一个问题", "第一个回答", "第二个问题"], "聊天消息顺序异常")
        _assert_true(chat_store.list_messages(session_id="chat-2", limit=10) == [], "不同session不应共享消息")

        anonymous_session = chat_store.ensure_session(session_id="chat-anonymous")
        _assert_true(anonymous_session.user_id == "", "匿名session初始不应绑定用户")
        claimed_session = chat_store.ensure_session(session_id="chat-anonymous", user_id="u1")
        _assert_true(claimed_session.user_id == "u1", "首次提供身份时应认领匿名session")
        try:
            chat_store.ensure_session(session_id="chat-anonymous", user_id="u2")
            raise AssertionError("其他用户不应恢复已归属的session")
        except ValueError:
            pass

        profile = chat_store.upsert_profile(
            user_id="u1",
            preferences={"output_style": "要点式"},
            confidence=0.9,
        )
        _assert_true(profile.preferences["output_style"] == "要点式", "用户偏好写入失败")
        _assert_true(chat_store.delete_profile(user_id="u1"), "用户偏好删除失败")
        _assert_true(chat_store.get_profile(user_id="u1") is None, "用户偏好删除后仍可读取")
        chat_store.close()

        expire_at = store.default_expiration()
        session_item_id = store.put(
            scope=MemoryScope.SESSION,
            text="Q: 用户问PE计算\nA: 已给出计算公式",
            user_id="u1",
            session_id="s1",
            tags=["turn"],
            payload={"route": "mcp_only"},
            importance=0.6,
            expires_at=expire_at,
        )
        _assert_true(bool(session_item_id), "会话记忆写入失败")

        user_item_id = store.put(
            scope=MemoryScope.USER,
            text="风险偏好=稳健; 投资期限=中长期",
            user_id="u1",
            session_id="",
            tags=["user_profile"],
            payload={"confidence": 0.9},
            importance=0.8,
            expires_at=expire_at,
        )
        _assert_true(bool(user_item_id), "用户记忆写入失败")

        project_item_id = store.put(
            scope=MemoryScope.PROJECT,
            text="项目结论：股份行估值中枢约13.5倍PE",
            user_id="",
            session_id="",
            tags=["project_fact"],
            payload={"source_count": 2},
            importance=0.7,
            expires_at=expire_at,
        )
        _assert_true(bool(project_item_id), "项目记忆写入失败")

        session_rows = store.list_recent(scope=MemoryScope.SESSION, limit=5, user_id="u1", session_id="s1")
        _assert_true(len(session_rows) == 1, "会话记忆读取数量异常")

        user_rows = store.recall(query="稳健 风险偏好", scope=MemoryScope.USER, top_k=3, user_id="u1", session_id="")
        _assert_true(len(user_rows) >= 1, "用户记忆召回失败")

        project_rows = store.recall(query="股份行 估值", scope=MemoryScope.PROJECT, top_k=3)
        _assert_true(len(project_rows) >= 1, "项目记忆召回失败")

        removed_none = store.prune_expired()
        _assert_true(removed_none == 0, "未过期数据不应被清理")

        removed_expired = store.prune_expired(now=expire_at + timedelta(seconds=1))
        _assert_true(removed_expired >= 3, "过期清理数量异常")

        left_rows = store.list_recent(scope=MemoryScope.PROJECT, limit=5)
        _assert_true(len(left_rows) == 0, "过期后不应残留项目记忆")
    finally:
        if store is not None:
            store.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    print("Memory存储测试通过：CRUD/召回/TTL行为符合预期。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
