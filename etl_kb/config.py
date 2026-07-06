from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except ImportError:  # Keep --mock runnable before dependencies are installed.
    def load_dotenv() -> bool:
        return False


@dataclass(frozen=True)
class Settings:
    feishu_app_id: str | None
    feishu_app_secret: str | None
    feishu_table_app_token: str | None
    feishu_table_id: str | None
    feishu_chat_id: str | None
    glm_api_key: str | None
    relay_api_key: str | None
    relay_base_url: str | None
    glm_chat_model: str
    glm_vision_model: str
    glm_embedding_model: str
    relay_vision_model: str
    database_path: Path
    image_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            feishu_app_id=os.getenv("FEISHU_APP_ID"),
            feishu_app_secret=os.getenv("FEISHU_APP_SECRET"),
            feishu_table_app_token=os.getenv("FEISHU_TABLE_APP_TOKEN"),
            feishu_table_id=os.getenv("FEISHU_TABLE_ID"),
            feishu_chat_id=os.getenv("FEISHU_CHAT_ID"),
            glm_api_key=os.getenv("GLM_API_KEY"),
            relay_api_key=os.getenv("RELAY_API_KEY"),
            relay_base_url=os.getenv("RELAY_BASE_URL"),
            glm_chat_model=os.getenv("GLM_CHAT_MODEL", "glm-4.5"),
            glm_vision_model=os.getenv("GLM_VISION_MODEL", "glm-4.5v"),
            glm_embedding_model=os.getenv("GLM_EMBEDDING_MODEL", "embedding-3"),
            relay_vision_model=os.getenv("RELAY_VISION_MODEL", "gpt-4o-mini"),
            database_path=Path(os.getenv("DATABASE_PATH", "data/knowledge.db")),
            image_dir=Path(os.getenv("IMAGE_DIR", "data/images")),
        )


class MissingConfigError(RuntimeError):
    pass


def require_env(settings: Settings, names: Iterable[str]) -> None:
    missing = [name for name in names if not getattr(settings, name.lower())]
    if missing:
        env_names = ", ".join(name.upper() for name in missing)
        raise MissingConfigError(f"Missing required environment variables: {env_names}")
