from __future__ import annotations

import argparse
import json
from pathlib import Path

from etl_kb.config import Settings
from etl_kb.llm import LLMClient
from etl_kb.models import ChatMessage
from etl_kb.pipeline import ETLPipeline
from etl_kb.storage import KnowledgeStore
from etl_kb.vehicle_normalizer import VehicleNormalizer


def load_messages(path: Path) -> list[ChatMessage]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return [
        ChatMessage(
            message_id=str(item["message_id"]),
            create_time=int(item["create_time"]),
            sender_id=item.get("sender_id"),
            msg_type=item.get("msg_type", "text"),
            content=item.get("content", {}),
            raw=item,
        )
        for item in items
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
