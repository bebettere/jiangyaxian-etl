from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests


class FeishuClient:
    BASE_URL = "https://open.feishu.cn/open-apis"

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        timeout: int = 20,
        max_retries: int = 3,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.timeout = timeout
        self.max_retries = max_retries
        self._tenant_access_token: str | None = None
        self.session = requests.Session()

    def tenant_access_token(self) -> str:
        if self._tenant_access_token:
            return self._tenant_access_token
        url = f"{self.BASE_URL}/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self.app_id, "app_secret": self.app_secret}
        data = self._request("POST", url, json=payload, auth=False).json()
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to get tenant_access_token: {data}")
        self._tenant_access_token = data["tenant_access_token"]
        return self._tenant_access_token

    def iter_bitable_records(
        self,
        app_token: str,
        table_id: str,
        page_size: int = 500,
    ) -> Iterator[dict[str, Any]]:
        page_token = None
        while True:
            params: dict[str, Any] = {"page_size": page_size}
            if page_token:
                params["page_token"] = page_token
            url = f"{self.BASE_URL}/bitable/v1/apps/{app_token}/tables/{table_id}/records"
            data = self._request("GET", url, params=params).json()
            self._ensure_feishu_ok(data, "list bitable records")
            payload = data.get("data", {})
            for item in payload.get("items", []):
                yield item
            if not payload.get("has_more"):
                break
            page_token = payload.get("page_token")

    def iter_chat_messages(
        self,
        chat_id: str,
        start_time: int | None = None,
        end_time: int | None = None,
        page_size: int = 50,
    ) -> Iterator[dict[str, Any]]:
        page_token = None
        while True:
            params: dict[str, Any] = {
                "container_id_type": "chat",
                "container_id": chat_id,
                "page_size": page_size,
            }
            if start_time:
                params["start_time"] = start_time
            if end_time:
                params["end_time"] = end_time
            if page_token:
                params["page_token"] = page_token
            url = f"{self.BASE_URL}/im/v1/messages"
            data = self._request("GET", url, params=params).json()
            self._ensure_feishu_ok(data, "list chat messages")
            payload = data.get("data", {})
            for item in payload.get("items", []):
                yield item
            if not payload.get("has_more"):
                break
            page_token = payload.get("page_token")

    def get_tmp_download_url(self, file_token: str) -> str | None:
        url = f"{self.BASE_URL}/drive/v1/medias/batch_get_tmp_download_url"
        data = self._request("POST", url, json={"file_tokens": [file_token]}).json()
        if data.get("code") != 0:
            return None
        items = data.get("data", {}).get("tmp_download_urls", [])
        if not items:
            return None
        return items[0].get("tmp_download_url")

    def download_image_token(self, token: str, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_url = self.get_tmp_download_url(token)
        if tmp_url:
            response = self._request("GET", tmp_url, auth=False, raw=True)
        else:
            url = f"{self.BASE_URL}/drive/v1/medias/{token}/download"
            response = self._request("GET", url, raw=True)

        suffix = self._suffix_from_response(response) or ".jpg"
        output = target_dir / f"{token}{suffix}"
        output.write_bytes(response.content)
        return output

    def _request(
        self,
        method: str,
        url: str,
        auth: bool = True,
        raw: bool = False,
        **kwargs: Any,
    ) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        if auth:
            headers["Authorization"] = f"Bearer {self.tenant_access_token()}"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    headers=headers,
                    **kwargs,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"retryable HTTP {response.status_code}", response=response)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                time.sleep(0.8 * attempt)
        raise RuntimeError("unreachable retry state")

    @staticmethod
    def _ensure_feishu_ok(data: dict[str, Any], action: str) -> None:
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to {action}: {data}")

    @staticmethod
    def _suffix_from_response(response: requests.Response) -> str | None:
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        suffix = mimetypes.guess_extension(content_type) if content_type else None
        if suffix:
            return suffix
        parsed = urlparse(response.url)
        suffix = Path(parsed.path).suffix
        return suffix or None


def parse_message_content(raw: dict[str, Any]) -> dict[str, Any]:
    content = raw.get("body", {}).get("content", "{}")
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"text": str(content)}
