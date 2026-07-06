from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[\s\-_·.]+", "", value).lower()


@dataclass(frozen=True)
class NormalizedVehicle:
    brand_std: str | None
    model_std: str | None
    year: int | None
    generation: str | None = None


class VehicleNormalizer:
    def __init__(self, alias_config: dict[str, Any]) -> None:
        self.alias_config = alias_config
        self.brand_aliases: dict[str, str] = {}
        self.model_aliases: dict[tuple[str, str], str] = {}
        self.generations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._build_indexes()

    @classmethod
    def from_file(cls, path: Path) -> "VehicleNormalizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def normalize(self, brand: str | None, model: str | None, year: int | str | None) -> NormalizedVehicle:
        parsed_year = self._parse_year(year)
        brand_std = self.brand_aliases.get(_clean(brand), brand.strip() if brand else None)
        model_std = None
        if model:
            model_key = _clean(model)
            if brand_std:
                model_std = self.model_aliases.get((brand_std, model_key))
            if not model_std:
                for (candidate_brand, candidate_model_key), candidate_model in self.model_aliases.items():
                    if candidate_model_key == model_key:
                        brand_std = brand_std or candidate_brand
                        model_std = candidate_model
                        break
            model_std = model_std or model.strip()
        generation = self._match_generation(brand_std, model_std, parsed_year)
        return NormalizedVehicle(brand_std, model_std, parsed_year, generation)

    def normalize_query_key(self, brand: str | None, model: str | None, year: int | str | None) -> str:
        normalized = self.normalize(brand, model, year)
        return "|".join(
            [
                normalized.brand_std or "",
                normalized.model_std or "",
                str(normalized.year or ""),
                normalized.generation or "",
            ]
        )

    def _build_indexes(self) -> None:
        for brand in self.alias_config.get("brands", []):
            brand_std = brand["std"]
            for alias in [brand_std, *brand.get("aliases", [])]:
                self.brand_aliases[_clean(alias)] = brand_std
            for model in brand.get("models", []):
                model_std = model["std"]
                for alias in [model_std, *model.get("aliases", [])]:
                    self.model_aliases[(brand_std, _clean(alias))] = model_std
                if model.get("generations"):
                    self.generations[(brand_std, model_std)] = model["generations"]

    def _match_generation(self, brand_std: str | None, model_std: str | None, year: int | None) -> str | None:
        if not brand_std or not model_std or not year:
            return None
        for generation in self.generations.get((brand_std, model_std), []):
            start = int(generation.get("start", 0))
            end = int(generation.get("end", 9999))
            if start <= year <= end:
                return generation.get("name")
        return None

    @staticmethod
    def _parse_year(year: int | str | None) -> int | None:
        if isinstance(year, int):
            return year
        if not year:
            return None
        match = re.search(r"(19|20)\d{2}", str(year))
        return int(match.group(0)) if match else None
