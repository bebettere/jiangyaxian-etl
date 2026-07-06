from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


class LLMClient:
    DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_CHAT_MODEL = "gpt-5.4"
    DEFAULT_VISION_MODEL = "gpt-5.4"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    OPENCLAW_CONFIG_PATH = Path("~/.openclaw/openclaw.json").expanduser()
    OPENCLAW_AUTH_PATH = Path("~/.openclaw/agents/main/agent/auth-profiles.json").expanduser()

    def __init__(
        self,
        openai_api_key: str | None,
        openai_base_url: str | None,
        openai_chat_model: str,
        openai_vision_model: str,
        openai_embedding_model: str,
        timeout: int = 60,
        max_retries: int = 2,
    ) -> None:
        api_key, base_url = self._load_openai_runtime(openai_api_key, openai_base_url)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat_model = self._resolve_model(openai_chat_model, self.DEFAULT_CHAT_MODEL)
        self.vision_model = self._resolve_model(openai_vision_model, self.DEFAULT_VISION_MODEL)
        self.embedding_model = self._resolve_model(openai_embedding_model, self.DEFAULT_EMBEDDING_MODEL)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def describe_image(self, image_path: Path) -> tuple[str, float]:
        self._require_openai()
        content = [
            {"type": "text", "text": "请识别图片中的汽车降压线/保险盒/接线位置，输出中文描述，并给出清晰度和置信度。"},
            {"type": "image_url", "image_url": {"url": self._data_url(image_path)}},
        ]
        data = self._chat(self.base_url, self.api_key, self.vision_model, content)
        text = self._message_text(data)
        confidence = self._extract_confidence(text)
        return text, confidence

    def extract_case(self, source_text: str, source_type: str, image_descriptions: list[str] | None = None) -> dict[str, Any]:
        self._require_openai()
        images = "\n".join(image_descriptions or [])
        prompt = f"""
你是行车记录仪降压线安装知识库的结构化抽取器。
请从输入中抽取 JSON，字段必须是：
brand, model, year, connection_description, quality_status, source_tags。
quality_status 只能取：未清洗、AI已解析、高风险勿自动回复。
如果车型或年份不确定，用 null。不要输出 Markdown，不要补充解释，只输出 JSON 对象。

来源类型：{source_type}
图片描述：
{images}

文本：
{source_text}
"""
        data = self._chat(
            self.base_url,
            self.api_key,
            self.chat_model,
            [{"type": "text", "text": prompt}],
        )
        return self._parse_json_object(self._message_text(data))

    def embed_text(self, text: str) -> list[float]:
        self._require_openai()
        url = f"{self.base_url}/embeddings"
        data = self._request(
            "POST",
            url,
            api_key=self.api_key,
            json={"model": self.embedding_model, "input": text},
        ).json()
        embedding = data.get("data", [{}])[0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Unexpected embedding response: {data}")
        return [float(x) for x in embedding]

    def _chat(self, base_url: str, api_key: str | None, model: str, content: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
        }
        return self._request("POST", url, api_key=api_key, json=payload).json()

    def _request(self, method: str, url: str, api_key: str | None, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"retryable HTTP {response.status_code}", response=response)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                time.sleep(1.0 * attempt)
        raise RuntimeError("unreachable retry state")

    def _require_openai(self) -> None:
        if not self.api_key:
            raise RuntimeError("未找到 OpenAI API key。请确认本机 OpenClaw 已配置 openai:default 认证，或设置 OPENAI_API_KEY。")

    @classmethod
    def _load_openai_runtime(cls, explicit_api_key: str | None, explicit_base_url: str | None) -> tuple[str | None, str]:
        api_key = cls._load_openclaw_api_key() or explicit_api_key
        base_url = cls._load_openclaw_base_url() or explicit_base_url or cls.DEFAULT_OPENAI_BASE_URL
        return api_key, base_url

    @classmethod
    def _load_openclaw_api_key(cls) -> str | None:
        if cls.OPENCLAW_AUTH_PATH.exists():
            try:
                data = json.loads(cls.OPENCLAW_AUTH_PATH.read_text(encoding="utf-8"))
                profiles = data.get("profiles", {})
                profile = profiles.get("openai:default", {})
                key = profile.get("key")
                if isinstance(key, str) and key.strip():
                    return key.strip()
            except (OSError, json.JSONDecodeError):
                pass
        return None

    @classmethod
    def _load_openclaw_base_url(cls) -> str | None:
        if cls.OPENCLAW_CONFIG_PATH.exists():
            try:
                data = json.loads(cls.OPENCLAW_CONFIG_PATH.read_text(encoding="utf-8"))
                base_url = (
                    data.get("models", {})
                    .get("providers", {})
                    .get("openai", {})
                    .get("baseUrl")
                )
                if isinstance(base_url, str) and base_url.strip():
                    return base_url.strip().rstrip("/")
            except (OSError, json.JSONDecodeError):
                pass
        return None

    @staticmethod
    def _resolve_model(requested: str | None, fallback: str) -> str:
        if not requested:
            return fallback
        normalized = requested.strip()
        if not normalized or normalized.startswith("glm") or normalized == "embedding-3":
            return fallback
        return normalized

    @staticmethod
    def _message_text(data: dict[str, Any]) -> str:
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return str(content)

    @staticmethod
    def _data_url(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/{mime};base64,{encoded}"

    @staticmethod
    def _extract_confidence(text: str) -> float:
        match = re.search(r"(?:置信度|confidence)[：:\s]*([01](?:\.\d+)?)", text, re.I)
        if match:
            return float(match.group(1))
        if "不清晰" in text or "无法识别" in text:
            return 0.35
        return 0.7

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.I | re.M).strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            cleaned = match.group(0)
        return json.loads(cleaned)
