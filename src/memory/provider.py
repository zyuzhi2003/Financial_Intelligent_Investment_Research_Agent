from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.memory.models import MemoryRecord, MemoryScope


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
