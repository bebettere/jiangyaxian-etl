from __future__ import annotations

import argparse
from pathlib import Path

from etl_kb.config import Settings
from etl_kb.feishu import FeishuClient
from etl_kb.llm import LLMClient
from etl_kb.pipeline import ETLPipeline
from etl_kb.storage import KnowledgeStore
from etl_kb.vehicle_normalizer import VehicleNormalizer


def build_pipeline(settings: Settings, mock: bool) -> ETLPipeline:
    feishu = None
    if settings.feishu_app_id and settings.feishu_app_secret:
        feishu = FeishuClient(settings.feishu_app_id, settings.feishu_app_secret)
    llm = LLMClient(
        glm_api_key=settings.glm_api_key,
        glm_chat_model=settings.glm_chat_model,
        glm_vision_model=settings.glm_vision_model,
        glm_embedding_model=settings.glm_embedding_model,
        relay_api_key=settings.relay_api_key,
        relay_base_url=settings.relay_base_url,
        relay_vision_model=settings.relay_vision_model,
    )
    normalizer = VehicleNormalizer.from_file(Path("configs/vehicle_aliases.json"))
    store = KnowledgeStore(settings.database_path)
    store.init_schema()
    return ETLPipeline(feishu, llm, normalizer, store, settings.image_dir, mock=mock)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ETL for hardwire-kit knowledge base.")
    parser.add_argument("--mock", action="store_true", help="Run with local mock records and no external API calls.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    table_parser = subparsers.add_parser("table", help="Process Feishu Base table records only.")
    table_parser.add_argument("--app-token", help="Override FEISHU_TABLE_APP_TOKEN.")
    table_parser.add_argument("--table-id", help="Override FEISHU_TABLE_ID.")

    chat_parser = subparsers.add_parser("chat", help="Process Feishu chat history only.")
    chat_parser.add_argument("--chat-id", help="Override FEISHU_CHAT_ID.")
    chat_parser.add_argument("--start-time", type=int, help="Message start time, Unix milliseconds.")
    chat_parser.add_argument("--end-time", type=int, help="Message end time, Unix milliseconds.")

    args = parser.parse_args()
    settings = Settings.from_env()
    pipeline = build_pipeline(settings, mock=args.mock)

    if args.command == "table":
        count = pipeline.run_table(
            app_token=args.app_token or settings.feishu_table_app_token,
            table_id=args.table_id or settings.feishu_table_id,
        )
    elif args.command == "chat":
        count = pipeline.run_chat(
            chat_id=args.chat_id or settings.feishu_chat_id,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    else:
        raise ValueError(args.command)
    print(f"ETL completed: {count} case(s) written to {settings.database_path}")


if __name__ == "__main__":
    main()
