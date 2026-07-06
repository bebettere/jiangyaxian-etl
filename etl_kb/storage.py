from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from .models import ExtractedCase


class KnowledgeStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand_std TEXT,
                model_std TEXT,
                year INTEGER,
                generation TEXT,
                connection_description TEXT NOT NULL,
                source TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                case_id INTEGER PRIMARY KEY,
                embedding_json TEXT NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES knowledge_cases(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_cases_vehicle
            ON knowledge_cases(brand_std, model_std, year, generation);
            """
        )
        self.conn.commit()

    def upsert_case(
        self,
        case: ExtractedCase,
        brand_std: str | None,
        model_std: str | None,
        year: int | None,
        generation: str | None,
        embedding: list[float] | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO knowledge_cases
            (brand_std, model_std, year, generation, connection_description, source, quality_status, evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                brand_std,
                model_std,
                year,
                generation,
                case.connection_description,
                case.source,
                case.quality_status,
                json.dumps(case.evidence, ensure_ascii=False),
            ),
        )
        case_id = int(cursor.lastrowid)
        if embedding is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(case_id, embedding_json, text) VALUES (?, ?, ?)",
                (case_id, json.dumps(embedding), case.connection_description),
            )
        self.conn.commit()
        return case_id

    def query_by_vehicle(self, brand_std: str | None, model_std: str | None, year: int | None) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM knowledge_cases
            WHERE (? IS NULL OR brand_std = ?)
              AND (? IS NULL OR model_std = ?)
              AND (? IS NULL OR year IS NULL OR year = ?)
            ORDER BY quality_status DESC, updated_at DESC
            """,
            (brand_std, brand_std, model_std, model_std, year, year),
        ).fetchall()
        return [dict(row) for row in rows]

    def search_similar(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT e.case_id, e.embedding_json, e.text, k.*
            FROM embeddings e
            JOIN knowledge_cases k ON k.id = e.case_id
            """
        ).fetchall()
        scored = []
        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query_embedding, embedding)
            item = dict(row)
            item["score"] = score
            scored.append(item)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)
