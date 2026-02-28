from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class InferenceSessionSnapshot:
    raw_rows: list[dict[str, Any]]
    filtered_rows: list[dict[str, Any]]
    meta: dict[str, Any]
    updated_at: str


_SNAPSHOT: InferenceSessionSnapshot | None = None


def set_inference_session_snapshot(
    *,
    raw_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    global _SNAPSHOT
    _SNAPSHOT = InferenceSessionSnapshot(
        raw_rows=[dict(r) for r in raw_rows if isinstance(r, dict)],
        filtered_rows=[dict(r) for r in filtered_rows if isinstance(r, dict)],
        meta=dict(meta),
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )


def get_inference_session_snapshot() -> InferenceSessionSnapshot | None:
    return _SNAPSHOT
