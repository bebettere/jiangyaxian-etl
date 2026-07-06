from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from etl_kb.config import Settings
from etl_kb.llm import LLMClient
from etl_kb.models import ChatMessage
from etl_kb.pipeline import ETLPipeline
from etl_kb.storage import KnowledgeStore
from etl_kb.vehicle_normalizer import VehicleNormalizer


def _parse_create_time(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def _parse_sender_id(item: dict[str, Any]) -> str | None:
    sender = item.get("sender")
    if isinstance(sender, dict):
        return sender.get("id")
    if isinstance(sender, str):
        return sender
    return item.get("sender_id")


def _parse_content(item: dict[str, Any]) -> dict[str, Any]:
    content = item.get("content")
    if isinstance(content, dict):
        return content
    return {"text": "" if content is None else str(content)}


def load_messages(path: Path) -> list[ChatMessage]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["messages"] if isinstance(data, dict) and "messages" in data else data
    return [
        ChatMessage(
            message_id=str(item["message_id"]),
            create_time=_parse_create_time(item["create_time"]),
            sender_id=_parse_sender_id(item),
            msg_type=item.get("msg_type", "text"),
            content=_parse_content(item),
            raw=item,
        )
        for item in items
        if not item.get("deleted")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a pre-exported chat history JSON (produced by the bot reading "
        "the group directly) into the knowledge base, running extraction/normalization/"
        "embedding exactly like `etl.py chat` but without calling the Feishu Open API."
    )
    parser.add_argument("json_path", type=Path, help="Path to the exported message JSON file.")
    args = parser.parse_args()

    settings = Settings.from_env()
    llm = LLMClient(
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
        openai_chat_model=settings.openai_chat_model,
        openai_vision_model=settings.openai_vision_model,
        openai_embedding_model=settings.openai_embedding_model,
    )
    normalizer = VehicleNormalizer.from_file(Path("configs/vehicle_aliases.json"))
    store = KnowledgeStore(settings.database_path)
    store.init_schema()
    pipeline = ETLPipeline(None, llm, normalizer, store, settings.image_dir, mock=False)

    messages = load_messages(args.json_path)
    count = pipeline.run_chat_from_messages(messages)
    print(f"Imported {count} case(s) from {args.json_path} into {settings.database_path}")


if __name__ == "__main__":
    main()
