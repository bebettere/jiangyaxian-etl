from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


class LLMClient:
    GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(
        self,
        glm_api_key: str | None,
        glm_chat_model: str,
        glm_vision_model: str,
        glm_embedding_model: str,
        relay_api_key: str | None = None,
        relay_base_url: str | None = None,
        relay_vision_model: str = "gpt-4o-mini",
        timeout: int = 60,
        max_retries: int = 2,
    ) -> None:
        self.glm_api_key = glm_api_key
        self.glm_chat_model = glm_chat_model
        self.glm_vision_model = glm_vision_model
        self.glm_embedding_model = glm_embedding_model
        self.relay_api_key = relay_api_key
        self.relay_base_url = relay_base_url.rstrip("/") if relay_base_url else None
        self.relay_vision_model = relay_vision_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def describe_image(self, image_path: Path) -> tuple[str, float]:
        self._require_glm()
        content = [
            {"type": "text", "text": "请识别图片中的汽车降压线/保险盒/接线位置，输出中文描述，并给出清晰度和置信度。"},
            {"type": "image_url", "image_url": {"url": self._data_url(image_path)}},
        ]
        data = self._chat(self.GLM_BASE_URL, self.glm_api_key, self.glm_vision_model, content)
        text = self._message_text(data)
        confidence = self._extract_confidence(text)
        if confidence < 0.55 and self.relay_api_key and self.relay_base_url:
            fallback = self._chat(self.relay_base_url, self.relay_api_key, self.relay_vision_model, content)
            return self._message_text(fallback), 0.6
        return text, confidence

    def extract_case(self, source_text: str, source_type: str, image_descriptions: list[str] | None = None) -> dict[str, Any]:
        self._require_glm()
        images = "\n".join(image_descriptions or [])
        prompt = f"""
你是行车记录仪降压线安装知识库的结构化抽取器。
请从输入中抽取 JSON，字段必须是：
brand, model, year, connection_description, quality_status, source_tags。
quality_status 只能取：未清洗、AI已解析、高风险勿自动回复。
如果车型或年份不确定，用 null。不要输出 Markdown。

来源类型：{source_type}
图片描述：
{images}

文本：
{source_text}
"""
        data = self._chat(
            self.GLM_BASE_URL,
            self.glm_api_key,
            self.glm_chat_model,
            [{"type": "text", "text": prompt}],
        )
        return self._parse_json_object(self._message_text(data))

    def embed_text(self, text: str) -> list[float]:
        self._require_glm()
        url = f"{self.GLM_BASE_URL}/embeddings"
        data = self._request(
            "POST",
            url,
            api_key=self.glm_api_key,
            json={"model": self.glm_embedding_model, "input": text},
        ).json()
        embedding = data.get("data", [{}])[0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Unexpected embedding response: {data}")
        return [float(x) for x in embedding]

    def _chat(self, base_url: str, api_key: str | None, model: str, content: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{base_url}/chat/completions"
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.1}
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

    def _require_glm(self) -> None:
        if not self.glm_api_key:
            raise RuntimeError("请先配置 GLM_API_KEY 环境变量。")

    @staticmethod
    def _message_text(data: dict[str, Any]) -> str:
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

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
