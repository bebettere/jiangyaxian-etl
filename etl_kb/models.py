from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TableRecord:
    record_id: str
    fields: dict[str, Any]


@dataclass
class ChatMessage:
    message_id: str
    create_time: int
    sender_id: str | None
    msg_type: str
    content: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseFragment:
    fragment_id: str
    messages: list[ChatMessage]
    text: str
    image_tokens: list[str]
    start_time: int
    end_time: int


@dataclass
class ExtractedCase:
    brand: str | None
    model: str | None
    year: int | None
    connection_description: str
    source: str
    quality_status: str
    evidence: dict[str, Any]
    image_descriptions: list[str] = field(default_factory=list)
