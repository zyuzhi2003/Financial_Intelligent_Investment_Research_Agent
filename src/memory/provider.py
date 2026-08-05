from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.memory.models import ChatMessage, ChatSession, MemoryRecord, MemoryScope, UserPreferenceProfile


class MemoryProvider(ABC):
    # 这个抽象类用于定义统一的记忆读写接口。

    @abstractmethod
    def put(
        self,
        *,
        scope: MemoryScope,
        text: str,
        user_id: str = "",
        session_id: str = "",
        tags: list[str] | None = None,
        payload: dict[str, Any] | None = None,
        importance: float = 0.5,
        expires_at: datetime | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def recall(
        self,
        *,
        query: str,
        scope: MemoryScope,
        top_k: int,
        user_id: str = "",
        session_id: str = "",
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_recent(
        self,
        *,
        scope: MemoryScope,
        limit: int,
        user_id: str = "",
        session_id: str = "",
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def prune_expired(self, *, now: datetime | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(
        self,
        *,
        scope: MemoryScope | None = None,
        user_id: str = "",
        session_id: str = "",
    ) -> int:
        raise NotImplementedError


class ChatMemoryProvider(ABC):
    # 这个抽象类用于定义聊天消息与用户偏好的持久化接口。

    @abstractmethod
    def ensure_session(self, *, session_id: str, user_id: str = "") -> ChatSession:
        raise NotImplementedError

    @abstractmethod
    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        user_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> ChatMessage:
        raise NotImplementedError

    @abstractmethod
    def list_messages(self, *, session_id: str, limit: int) -> list[ChatMessage]:
        raise NotImplementedError

    @abstractmethod
    def get_profile(self, *, user_id: str) -> UserPreferenceProfile | None:
        raise NotImplementedError

    @abstractmethod
    def upsert_profile(
        self,
        *,
        user_id: str,
        preferences: dict[str, Any],
        confidence: float,
    ) -> UserPreferenceProfile:
        raise NotImplementedError

    @abstractmethod
    def delete_profile(self, *, user_id: str) -> bool:
        raise NotImplementedError
