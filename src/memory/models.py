from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryScope(str, Enum):
    # 这个枚举用于标识三类记忆作用域。
    SESSION = "session"
    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True)
class MemoryRecord:
    # 这个字段用于标识记忆记录主键。
    item_id: str
    # 这个字段用于标识记忆作用域。
    scope: MemoryScope
    # 这个字段用于存储记忆文本内容。
    text: str
    # 这个字段用于标识用户ID。
    user_id: str
    # 这个字段用于标识会话ID。
    session_id: str
    # 这个字段用于记录标签集合。
    tags: list[str]
    # 这个字段用于记录附加结构化信息。
    payload: dict[str, Any]
    # 这个字段用于记录记忆重要度。
    importance: float
    # 这个字段用于记录创建时间。
    created_at: datetime
    # 这个字段用于记录更新时间。
    updated_at: datetime
    # 这个字段用于记录失效时间。
    expires_at: datetime | None


@dataclass(frozen=True)
class UserProfile:
    # 这个字段用于表示风险偏好。
    risk_preference: str
    # 这个字段用于表示投资期限偏好。
    investment_horizon: str
    # 这个字段用于表示输出风格偏好。
    output_style: str
    # 这个字段用于表示关注主题。
    focus_topics: list[str]
    # 这个字段用于表示语言偏好。
    language_preference: str
    # 这个字段用于表示提取置信度。
    confidence: float


@dataclass(frozen=True)
class MemoryTrace:
    # 这个字段用于记录命中记忆条数。
    recalled_count: int
    # 这个字段用于记录写回结果。
    write_status: str
    # 这个字段用于记录命中的作用域分布。
    scope_counts: dict[str, int]


@dataclass(frozen=True)
class ChatSession:
    # 这个结构用于描述可恢复的聊天会话。
    session_id: str
    user_id: str
    summary: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ChatMessage:
    # 这个结构用于保存一条有序的聊天消息。
    message_id: str
    session_id: str
    user_id: str
    sequence_no: int
    role: str
    content: str
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class UserPreferenceProfile:
    # 这个结构用于保存跨聊天会话生效的用户偏好。
    user_id: str
    preferences: dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
