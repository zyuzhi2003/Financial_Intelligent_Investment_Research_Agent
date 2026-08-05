from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from langchain_chroma import Chroma

from src.config import Settings
from src.memory.models import ChatMessage, ChatSession, MemoryRecord, MemoryScope, UserPreferenceProfile
from src.memory.provider import ChatMemoryProvider, MemoryProvider
from src.rag.embeddings import create_embeddings


def _utc_now() -> datetime:
    # 这个函数用于统一生成UTC时间戳。
    return datetime.now(timezone.utc)


def _normalize_identity(value: str | None) -> str:
    # 这个函数用于标准化身份字段，避免空值分支膨胀。
    return (value or "").strip()


@dataclass(frozen=True)
class SqlWhere:
    # 这个结构用于组织SQL过滤条件与参数。
    clause: str
    params: tuple[object, ...]


class ChatMemoryStore(ChatMemoryProvider):
    # 这个类用于通过SQLite持久化有序聊天消息和用户偏好，不依赖向量模型。
    def __init__(self, memory_dir: Path) -> None:
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(memory_dir / "chat_memory.sqlite3"))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # 这个函数用于无破坏地初始化聊天记忆表和索引。
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',
                sequence_no INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(session_id, sequence_no),
                FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_sequence
                ON chat_messages(session_id, sequence_no DESC);

            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferences_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> ChatSession:
        # 这个函数用于把数据库行转换为聊天会话模型。
        return ChatSession(
            session_id=str(row["session_id"]),
            user_id=str(row["user_id"]),
            summary=str(row["summary"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ChatMessage:
        # 这个函数用于把数据库行转换为聊天消息模型。
        return ChatMessage(
            message_id=str(row["message_id"]),
            session_id=str(row["session_id"]),
            user_id=str(row["user_id"]),
            sequence_no=int(row["sequence_no"]),
            role=str(row["role"]),
            content=str(row["content"]),
            payload=dict(json.loads(str(row["payload_json"]))),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> UserPreferenceProfile:
        # 这个函数用于把数据库行转换为用户偏好模型。
        return UserPreferenceProfile(
            user_id=str(row["user_id"]),
            preferences=dict(json.loads(str(row["preferences_json"]))),
            confidence=float(row["confidence"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    def ensure_session(self, *, session_id: str, user_id: str = "") -> ChatSession:
        # 这个方法用于创建会话或验证已有会话的用户归属。
        normalized_session = session_id.strip()
        normalized_user = user_id.strip()
        if not normalized_session:
            raise ValueError("session_id不能为空。")

        row = self._conn.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?",
            (normalized_session,),
        ).fetchone()
        if row is not None:
            session = self._row_to_session(row)
            if session.user_id and normalized_user and session.user_id != normalized_user:
                raise ValueError("session_id已属于其他用户。")
            if not session.user_id and normalized_user:
                # 这个分支用于让首次提供身份的用户安全认领匿名会话。
                now = _utc_now().isoformat()
                self._conn.execute(
                    "UPDATE chat_sessions SET user_id = ?, updated_at = ? WHERE session_id = ? AND user_id = ''",
                    (normalized_user, now, normalized_session),
                )
                self._conn.commit()
                claimed = self._conn.execute(
                    "SELECT * FROM chat_sessions WHERE session_id = ?",
                    (normalized_session,),
                ).fetchone()
                return self._row_to_session(claimed)
            return session

        now = _utc_now().isoformat()
        self._conn.execute(
            "INSERT INTO chat_sessions(session_id, user_id, summary, created_at, updated_at) VALUES(?, ?, '', ?, ?)",
            (normalized_session, normalized_user, now, now),
        )
        self._conn.commit()
        created = self._conn.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?",
            (normalized_session,),
        ).fetchone()
        return self._row_to_session(created)

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        user_id: str = "",
        payload: dict | None = None,
    ) -> ChatMessage:
        # 这个方法用于原子追加一条具有稳定顺序号的聊天消息。
        normalized_role = role.strip().lower()
        normalized_content = content.strip()
        if normalized_role not in {"user", "assistant"}:
            raise ValueError("role仅支持user/assistant。")
        if not normalized_content:
            raise ValueError("消息内容不能为空。")
        session = self.ensure_session(session_id=session_id, user_id=user_id)
        now = _utc_now().isoformat()
        message_id = str(uuid.uuid4())
        with self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_no FROM chat_messages WHERE session_id = ?",
                (session.session_id,),
            ).fetchone()
            sequence_no = int(row["next_no"])
            self._conn.execute(
                """
                INSERT INTO chat_messages(
                    message_id, session_id, user_id, sequence_no, role, content, payload_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session.session_id,
                    session.user_id,
                    sequence_no,
                    normalized_role,
                    normalized_content,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session.session_id),
            )
        created = self._conn.execute(
            "SELECT * FROM chat_messages WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        return self._row_to_message(created)

    def list_messages(self, *, session_id: str, limit: int) -> list[ChatMessage]:
        # 这个方法用于读取最近消息并恢复为从旧到新的顺序。
        rows = self._conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE session_id = ?
            ORDER BY sequence_no DESC
            LIMIT ?
            """,
            (session_id.strip(), max(1, limit)),
        ).fetchall()
        return [self._row_to_message(row) for row in reversed(rows)]

    def get_profile(self, *, user_id: str) -> UserPreferenceProfile | None:
        # 这个方法用于读取指定用户当前生效的偏好画像。
        normalized_user = user_id.strip()
        if not normalized_user:
            return None
        row = self._conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (normalized_user,),
        ).fetchone()
        return self._row_to_profile(row) if row is not None else None

    def upsert_profile(
        self,
        *,
        user_id: str,
        preferences: dict,
        confidence: float,
    ) -> UserPreferenceProfile:
        # 这个方法用于创建或覆盖指定用户的结构化偏好画像。
        normalized_user = user_id.strip()
        if not normalized_user:
            raise ValueError("user_id不能为空。")
        now = _utc_now().isoformat()
        clamped_confidence = max(0.0, min(1.0, float(confidence)))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO user_profiles(user_id, preferences_json, confidence, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferences_json = excluded.preferences_json,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_user,
                    json.dumps(preferences, ensure_ascii=False),
                    clamped_confidence,
                    now,
                    now,
                ),
            )
        profile = self.get_profile(user_id=normalized_user)
        if profile is None:
            raise RuntimeError("用户偏好写入失败。")
        return profile

    def delete_profile(self, *, user_id: str) -> bool:
        # 这个方法用于删除指定用户的全部长期偏好。
        cursor = self._conn.execute(
            "DELETE FROM user_profiles WHERE user_id = ?",
            (user_id.strip(),),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        # 这个方法用于释放聊天记忆数据库连接。
        self._conn.close()


class HybridMemoryStore(MemoryProvider):
    # 这个类用于实现SQLite主存+Chroma语义召回的混合记忆存储。
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._memory_dir = settings.memory_dir
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        self._sqlite_path = self._memory_dir / "memory.sqlite3"
        self._conn = sqlite3.connect(str(self._sqlite_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

        self._vectorstore: Chroma | None = None
        self._embedding_error: str = ""
        if settings.memory_backend in {"hybrid", "chroma"}:
            try:
                embedding = create_embeddings(settings)
                chroma_dir = settings.memory_dir / "chroma"
                chroma_dir.mkdir(parents=True, exist_ok=True)
                self._vectorstore = Chroma(
                    collection_name=settings.memory_collection,
                    embedding_function=embedding,
                    persist_directory=str(chroma_dir),
                )
            except Exception as exc:  # noqa: BLE001
                self._embedding_error = str(exc)

    def _ensure_schema(self) -> None:
        # 这个函数用于初始化记忆表结构与索引。
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                item_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                payload_json TEXT NOT NULL DEFAULT '{}',
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_events (
                event_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_memory_scope_user_session
                ON memory_items(scope, user_id, session_id, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_expires_at
                ON memory_items(expires_at);
            """
        )
        self._conn.commit()

    def _make_where(
        self,
        *,
        scope: MemoryScope | None,
        user_id: str,
        session_id: str,
    ) -> SqlWhere:
        # 这个函数用于按scope/user/session动态构造过滤条件。
        clauses: list[str] = []
        params: list[object] = []

        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope.value)

        normalized_user = _normalize_identity(user_id)
        normalized_session = _normalize_identity(session_id)

        if normalized_user:
            clauses.append("user_id = ?")
            params.append(normalized_user)
        if normalized_session:
            clauses.append("session_id = ?")
            params.append(normalized_session)

        where = " AND ".join(clauses) if clauses else "1=1"
        return SqlWhere(clause=where, params=tuple(params))

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        # 这个函数用于把SQLite行对象转换为MemoryRecord。
        expires_at_raw = row["expires_at"]
        expires_at = datetime.fromisoformat(expires_at_raw) if expires_at_raw else None
        return MemoryRecord(
            item_id=str(row["item_id"]),
            scope=MemoryScope(str(row["scope"])),
            text=str(row["text"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            tags=list(json.loads(str(row["tags_json"]))),
            payload=dict(json.loads(str(row["payload_json"]))),
            importance=float(row["importance"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            expires_at=expires_at,
        )

    def put(
        self,
        *,
        scope: MemoryScope,
        text: str,
        user_id: str = "",
        session_id: str = "",
        tags: list[str] | None = None,
        payload: dict | None = None,
        importance: float = 0.5,
        expires_at: datetime | None = None,
    ) -> str:
        # 这个方法用于写入一条新记忆，并尽量同步到语义索引层。
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("记忆内容不能为空。")

        item_id = str(uuid.uuid4())
        now = _utc_now().isoformat()
        normalized_user = _normalize_identity(user_id)
        normalized_session = _normalize_identity(session_id)
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        expires_at_iso = expires_at.isoformat() if expires_at else None

        self._conn.execute(
            """
            INSERT INTO memory_items(
                item_id, scope, user_id, session_id, text, tags_json,
                payload_json, importance, created_at, updated_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                scope.value,
                normalized_user,
                normalized_session,
                normalized_text,
                tags_json,
                payload_json,
                float(max(0.0, min(1.0, importance))),
                now,
                now,
                expires_at_iso,
            ),
        )
        self._conn.execute(
            """
            INSERT INTO memory_events(event_id, item_id, action, created_at, details_json)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                item_id,
                "insert",
                now,
                json.dumps({"scope": scope.value}, ensure_ascii=False),
            ),
        )
        self._conn.commit()

        if self._vectorstore is not None:
            try:
                self._vectorstore.add_texts(
                    texts=[normalized_text],
                    metadatas=[
                        {
                            "item_id": item_id,
                            "scope": scope.value,
                            "user_id": normalized_user,
                            "session_id": normalized_session,
                            "importance": float(max(0.0, min(1.0, importance))),
                        }
                    ],
                    ids=[item_id],
                )
            except Exception:  # noqa: BLE001
                # 这个异常分支用于保证语义层异常不阻塞主存写入。
                pass

        return item_id

    def _active_rows_by_ids(self, ids: list[str]) -> list[MemoryRecord]:
        # 这个函数用于按ID批量读取有效记忆记录。
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        now_iso = _utc_now().isoformat()
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_items
            WHERE item_id IN ({placeholders})
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            tuple(ids) + (now_iso,),
        ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        order_map = {value: idx for idx, value in enumerate(ids)}
        records.sort(key=lambda record: order_map.get(record.item_id, 10**9))
        return records

    def recall(
        self,
        *,
        query: str,
        scope: MemoryScope,
        top_k: int,
        user_id: str = "",
        session_id: str = "",
    ) -> list[MemoryRecord]:
        # 这个方法用于召回指定作用域的记忆，优先语义召回，失败回退最近记录。
        normalized_user = _normalize_identity(user_id)
        normalized_session = _normalize_identity(session_id)
        limit = max(1, top_k)

        if self._vectorstore is not None and query.strip():
            try:
                filters: dict[str, object] = {"scope": scope.value}
                if normalized_user:
                    filters["user_id"] = normalized_user
                if normalized_session:
                    filters["session_id"] = normalized_session
                docs = self._vectorstore.similarity_search(
                    query=query,
                    k=limit,
                    filter=filters,
                )
                candidate_ids: list[str] = []
                for doc in docs:
                    item_id = str(doc.metadata.get("item_id", "")).strip()
                    if item_id:
                        candidate_ids.append(item_id)
                records = self._active_rows_by_ids(candidate_ids)
                if records:
                    return records
            except Exception:  # noqa: BLE001
                pass

        return self.list_recent(
            scope=scope,
            limit=limit,
            user_id=normalized_user,
            session_id=normalized_session,
        )

    def list_recent(
        self,
        *,
        scope: MemoryScope,
        limit: int,
        user_id: str = "",
        session_id: str = "",
    ) -> list[MemoryRecord]:
        # 这个方法用于读取最近写入且未过期的记忆。
        where = self._make_where(scope=scope, user_id=user_id, session_id=session_id)
        now_iso = _utc_now().isoformat()
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_items
            WHERE {where.clause}
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            where.params + (now_iso, max(1, limit)),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def prune_expired(self, *, now: datetime | None = None) -> int:
        # 这个方法用于清理过期记忆并返回清理条数。
        now_iso = (now or _utc_now()).isoformat()
        rows = self._conn.execute(
            "SELECT item_id FROM memory_items WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now_iso,),
        ).fetchall()
        item_ids = [str(row["item_id"]) for row in rows]
        if not item_ids:
            return 0

        placeholders = ",".join("?" for _ in item_ids)
        self._conn.execute(f"DELETE FROM memory_items WHERE item_id IN ({placeholders})", tuple(item_ids))
        self._conn.commit()

        if self._vectorstore is not None:
            try:
                self._vectorstore.delete(ids=item_ids)
            except Exception:  # noqa: BLE001
                pass

        return len(item_ids)

    def clear(
        self,
        *,
        scope: MemoryScope | None = None,
        user_id: str = "",
        session_id: str = "",
    ) -> int:
        # 这个方法用于按作用域或身份过滤清空记忆。
        where = self._make_where(scope=scope, user_id=user_id, session_id=session_id)
        rows = self._conn.execute(
            f"SELECT item_id FROM memory_items WHERE {where.clause}",
            where.params,
        ).fetchall()
        item_ids = [str(row["item_id"]) for row in rows]
        if not item_ids:
            return 0

        self._conn.execute(f"DELETE FROM memory_items WHERE {where.clause}", where.params)
        self._conn.commit()

        if self._vectorstore is not None:
            try:
                self._vectorstore.delete(ids=item_ids)
            except Exception:  # noqa: BLE001
                pass

        return len(item_ids)

    def default_expiration(self) -> datetime:
        # 这个方法用于返回默认TTL对应的过期时间。
        return _utc_now() + timedelta(days=max(1, self._settings.memory_ttl_days))

    @property
    def embedding_error(self) -> str:
        # 这个属性用于暴露语义层初始化失败原因。
        return self._embedding_error

    def close(self) -> None:
        # 这个方法用于显式关闭SQLite连接，便于测试和资源回收。
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
