from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .feishu import FeishuClient, parse_message_content
from .llm import LLMClient
from .models import ChatMessage, ExtractedCase, TableRecord
from .segmenter import segment_chat_messages
from .storage import KnowledgeStore
from .vehicle_normalizer import VehicleNormalizer


TABLE_FIELD_ALIASES = {
    "ticket_id": ["工单号", "ticket_id", "Ticket ID"],
    "brand": ["汽车品牌", "品牌", "brand"],
    "model": ["车型", "model"],
    "year": ["年份", "年款", "year"],
    "connection": ["连接说明", "接线说明", "连接位置描述", "connection"],
    "images": ["方案配图", "图片", "images"],
    "message_link": ["飞书消息链接", "消息链接", "message_link"],
    "message_id": ["消息ID", "message_id"],
    "reply_status": ["回复状态", "reply_status"],
}


class ETLPipeline:
    def __init__(
        self,
        feishu: FeishuClient | None,
        llm: LLMClient,
        normalizer: VehicleNormalizer,
        store: KnowledgeStore,
        image_dir: Path,
        mock: bool = False,
    ) -> None:
        self.feishu = feishu
        self.llm = llm
        self.normalizer = normalizer
        self.store = store
        self.image_dir = image_dir
        self.mock = mock

    def run_table(self, app_token: str | None = None, table_id: str | None = None) -> int:
        records = self._mock_table_records() if self.mock else self._fetch_table_records(app_token, table_id)
        count = 0
        for record in records:
            extracted = self._extract_from_table(record)
            self._write_case(extracted)
            count += 1
        return count

    def run_chat(self, chat_id: str | None = None, start_time: int | None = None, end_time: int | None = None) -> int:
        messages = self._mock_chat_messages() if self.mock else self._fetch_chat_messages(chat_id, start_time, end_time)
        fragments = segment_chat_messages(messages)
        count = 0
        for fragment in fragments:
            image_descriptions = self._describe_images(fragment.image_tokens)
            if self.mock:
                data = {
                    "brand": "Toyota",
                    "model": "Prius",
                    "year": 2018,
                    "connection_description": "ACC 接点使用保险盒内点烟器保险，常电使用室内灯保险，地线接车身螺丝。",
                    "quality_status": "AI已解析",
                    "source_tags": ["mock", "群聊"],
                }
            else:
                data = self.llm.extract_case(fragment.text, "群聊", image_descriptions)
            extracted = self._case_from_llm_data(
                data,
                source="群聊",
                evidence={
                    "fragment_id": fragment.fragment_id,
                    "message_ids": [item.message_id for item in fragment.messages],
                    "start_time": fragment.start_time,
                    "end_time": fragment.end_time,
                    "image_tokens": fragment.image_tokens,
                },
                image_descriptions=image_descriptions,
            )
            self._write_case(extracted)
            count += 1
        return count

    def _fetch_table_records(self, app_token: str | None, table_id: str | None) -> list[TableRecord]:
        if not self.feishu or not app_token or not table_id:
            raise RuntimeError("请先配置 FEISHU_APP_ID/SECRET、FEISHU_TABLE_APP_TOKEN、FEISHU_TABLE_ID，或使用 --mock。")
        return [
            TableRecord(record_id=item["record_id"], fields=item.get("fields", {}))
            for item in self.feishu.iter_bitable_records(app_token, table_id)
        ]

    def _fetch_chat_messages(self, chat_id: str | None, start_time: int | None, end_time: int | None) -> list[ChatMessage]:
        if not self.feishu or not chat_id:
            raise RuntimeError("请先配置 FEISHU_APP_ID/SECRET、FEISHU_CHAT_ID，或使用 --mock。")
        messages = []
        for item in self.feishu.iter_chat_messages(chat_id, start_time, end_time):
            messages.append(
                ChatMessage(
                    message_id=item.get("message_id", ""),
                    create_time=int(item.get("create_time", 0)),
                    sender_id=item.get("sender", {}).get("id"),
                    msg_type=item.get("msg_type", ""),
                    content=parse_message_content(item),
                    raw=item,
                )
            )
        return messages

    def _extract_from_table(self, record: TableRecord) -> ExtractedCase:
        brand = self._field(record.fields, "brand")
        model = self._field(record.fields, "model")
        year = self._field(record.fields, "year")
        connection = self._field(record.fields, "connection") or ""
        image_tokens = self._extract_tokens(self._field(record.fields, "images"))
        image_descriptions = self._describe_images(image_tokens)
        if self.mock:
            data = {
                "brand": brand,
                "model": model,
                "year": year,
                "connection_description": connection,
                "quality_status": "人工已确认",
                "source_tags": ["mock", "飞书表格"],
            }
        else:
            source_text = json.dumps(record.fields, ensure_ascii=False)
            data = self.llm.extract_case(source_text, "飞书表格", image_descriptions)
            data.setdefault("brand", brand)
            data.setdefault("model", model)
            data.setdefault("year", year)
            data.setdefault("connection_description", connection)
        return self._case_from_llm_data(
            data,
            source="飞书表格",
            evidence={
                "record_id": record.record_id,
                "ticket_id": self._field(record.fields, "ticket_id"),
                "message_link": self._field(record.fields, "message_link"),
                "message_id": self._field(record.fields, "message_id"),
                "reply_status": self._field(record.fields, "reply_status"),
                "image_tokens": image_tokens,
            },
            image_descriptions=image_descriptions,
        )

    def _write_case(self, extracted: ExtractedCase) -> int:
        normalized = self.normalizer.normalize(extracted.brand, extracted.model, extracted.year)
        if self.mock:
            seed = hashlib.sha1(extracted.connection_description.encode("utf-8")).digest()
            embedding = [float(byte) / 255.0 for byte in seed[:16]]
        else:
            embedding = self.llm.embed_text(
                f"{normalized.brand_std or ''} {normalized.model_std or ''} {normalized.year or ''} "
                f"{extracted.connection_description}"
            )
        return self.store.upsert_case(
            extracted,
            normalized.brand_std,
            normalized.model_std,
            normalized.year,
            normalized.generation,
            embedding,
        )

    def _describe_images(self, tokens: list[str]) -> list[str]:
        descriptions = []
        for token in tokens:
            if self.mock:
                descriptions.append(f"mock image {token}: 保险盒位置清晰，可见 ACC 和常电取电点。")
                continue
            if not self.feishu:
                continue
            image_path = self.feishu.download_image_token(token, self.image_dir)
            text, confidence = self.llm.describe_image(image_path)
            descriptions.append(f"{text}\nconfidence={confidence}")
        return descriptions

    @staticmethod
    def _field(fields: dict[str, Any], canonical: str) -> Any:
        for name in TABLE_FIELD_ALIASES[canonical]:
            if name in fields:
                value = fields[name]
                if isinstance(value, list) and len(value) == 1:
                    return value[0]
                return value
        return None

    @staticmethod
    def _extract_tokens(value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            for key in ("file_token", "token", "file_key", "image_key"):
                if value.get(key):
                    return [str(value[key])]
        if isinstance(value, list):
            tokens = []
            for item in value:
                tokens.extend(ETLPipeline._extract_tokens(item))
            return tokens
        return []

    @staticmethod
    def _case_from_llm_data(
        data: dict[str, Any],
        source: str,
        evidence: dict[str, Any],
        image_descriptions: list[str],
    ) -> ExtractedCase:
        return ExtractedCase(
            brand=data.get("brand"),
            model=data.get("model"),
            year=data.get("year"),
            connection_description=data.get("connection_description") or "",
            source=source,
            quality_status=data.get("quality_status") or "AI已解析",
            evidence=evidence,
            image_descriptions=image_descriptions,
        )

    @staticmethod
    def _mock_table_records() -> list[TableRecord]:
        return [
            TableRecord(
                record_id="mock_rec_001",
                fields={
                    "工单号": "MOCK-001",
                    "汽车品牌": "丰田",
                    "车型": "普锐斯",
                    "年份": "2018",
                    "连接说明": "ACC 接点接点烟器保险，常电接室内灯保险，地线接左侧金属螺丝。",
                    "方案配图": [{"file_token": "mock_image_001"}],
                    "飞书消息链接": "https://example.com/mock",
                    "回复状态": "已回复",
                },
            )
        ]

    @staticmethod
    def _mock_chat_messages() -> list[ChatMessage]:
        return [
            ChatMessage("m1", 1710000000000, "u1", "text", {"text": "2018 Prius 降压线怎么接？"}),
            ChatMessage("m2", 1710000300000, "u2", "text", {"text": "ACC 用点烟器保险，常电用室内灯保险，地线接车身螺丝。"}),
            ChatMessage("m3", 1710000400000, "u2", "image", {"image_key": "mock_image_002"}),
        ]
