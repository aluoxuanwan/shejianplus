from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class InferenceSessionExportResult:
    output_dir: Path
    raw_csv: Path
    filtered_csv: Path
    meta_json: Path
    raw_rows: int
    filtered_rows: int


def export_inference_session_timeseries(
    *,
    export_root: Path,
    session_meta: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
) -> InferenceSessionExportResult:
    export_root = Path(export_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = export_root / f"inference_session_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "raw_2d_timeseries.csv"
    filtered_csv = out_dir / "filtered_2d_timeseries.csv"
    meta_json = out_dir / "session_meta.json"

    fieldnames = _timeseries_fieldnames()
    _write_csv(raw_csv, raw_rows, fieldnames)
    _write_csv(filtered_csv, filtered_rows, fieldnames)

    meta_payload = {
        "schema_version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "session_meta": session_meta,
        "summary": {
            "raw_rows": len(raw_rows),
            "filtered_rows": len(filtered_rows),
        },
    }
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    return InferenceSessionExportResult(
        output_dir=out_dir,
        raw_csv=raw_csv,
        filtered_csv=filtered_csv,
        meta_json=meta_json,
        raw_rows=len(raw_rows),
        filtered_rows=len(filtered_rows),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _timeseries_fieldnames() -> list[str]:
    return [
        "frame_idx",
        "video_time_s",
        "wall_time_s",
        "target",
        "keypoint_name",
        "x_px",
        "y_px",
        "x_cm",
        "y_cm",
        "score",
        "src_fps",
        "display_filter",
        "row_kind",
    ]
