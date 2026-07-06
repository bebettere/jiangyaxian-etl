from __future__ import annotations

import hashlib
import re
from collections import Counter

from .models import CaseFragment, ChatMessage


TOPIC_KEYWORDS = {
    "保险盒",
    "acc",
    "常电",
    "地线",
    "降压线",
    "取电",
    "行车记录仪",
    "fuse",
    "hardwire",
    "parking",
}


def segment_chat_messages(
    messages: list[ChatMessage],
    window_minutes: int = 240,
    min_topic_score: int = 1,
) -> list[CaseFragment]:
    if not messages:
        return []
    ordered = sorted(messages, key=lambda item: item.create_time)
    fragments: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    last_time: int | None = None
    window_ms = window_minutes * 60 * 1000

    for message in ordered:
        gap_exceeded = last_time is not None and message.create_time - last_time > window_ms
        topic_shift = bool(current) and not _is_related(current, message)
        if current and (gap_exceeded or topic_shift):
            fragments.append(current)
            current = []
        current.append(message)
        last_time = message.create_time
    if current:
        fragments.append(current)

    return [_to_fragment(chunk) for chunk in fragments if _topic_score(chunk) >= min_topic_score]


def _is_related(current: list[ChatMessage], message: ChatMessage) -> bool:
    if _message_topic_score(message) > 0:
        return True
    existing_terms = _terms(" ".join(_message_text(item) for item in current))
    new_terms = _terms(_message_text(message))
    if not new_terms:
        return True
    overlap = set(existing_terms) & set(new_terms)
    return bool(overlap)


def _to_fragment(messages: list[ChatMessage]) -> CaseFragment:
    text = "\n".join(
        f"[{_speaker_label(item)}] {_message_text(item)}" for item in messages if _message_text(item)
    )
    image_tokens: list[str] = []
    for message in messages:
        image_tokens.extend(_image_tokens(message))
    raw_id = "|".join(item.message_id for item in messages)
    fragment_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
    return CaseFragment(
        fragment_id=fragment_id,
        messages=messages,
        text=text,
        image_tokens=image_tokens,
        start_time=messages[0].create_time,
        end_time=messages[-1].create_time,
    )


def _topic_score(messages: list[ChatMessage]) -> int:
    return sum(_message_topic_score(message) for message in messages)


def _message_topic_score(message: ChatMessage) -> int:
    text = _message_text(message).lower()
    return sum(1 for keyword in TOPIC_KEYWORDS if keyword in text)


def _message_text(message: ChatMessage) -> str:
    content = message.content
    if "text" in content:
        return str(content["text"])
    if "title" in content:
        return str(content["title"])
    return str(content) if message.msg_type == "text" else ""


def _speaker_label(message: ChatMessage) -> str:
    sender = message.raw.get("sender") if isinstance(message.raw, dict) else None
    if isinstance(sender, dict) and sender.get("name"):
        return str(sender["name"])
    return message.sender_id or "unknown"


def _image_tokens(message: ChatMessage) -> list[str]:
    content = message.content
    candidates = [
        content.get("image_key"),
        content.get("file_key"),
        content.get("file_token"),
        content.get("token"),
    ]
    return [str(item) for item in candidates if item]


def _terms(text: str) -> Counter[str]:
    words = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}", text.lower())
    return Counter(words)
