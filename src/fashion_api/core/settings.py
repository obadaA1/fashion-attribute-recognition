from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_root: Path
    cors_origins: tuple[str, ...]
    max_image_bytes: int
    request_timeout_seconds: int
    environment: str


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings(
        model_root=Path(os.getenv("FASHION_MODEL_ROOT", "/models/fashion/current")),
        cors_origins=_csv_env(
            "FASHION_CORS_ORIGINS",
            "https://obadaalsehli.com,http://localhost:3000,http://127.0.0.1:3000",
        ),
        max_image_bytes=int(os.getenv("FASHION_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
        request_timeout_seconds=int(os.getenv("FASHION_REQUEST_TIMEOUT_SECONDS", "30")),
        environment=os.getenv("FASHION_ENV", "production"),
    )

