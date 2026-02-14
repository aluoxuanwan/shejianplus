from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2

from archery_plus.pipelines.auto_annotator_halpe26 import HALPE26_NAMES
from archery_plus.pipelines.auto_annotator_rtmo_archery import ARCHERY_KEYPOINTS


@dataclass
class BuildDatasetResult:
    total_records: int
    train_records: int
    val_records: int
    skipped_records: int
    output_dir: Path


def build_training_dataset(
    project_root: Path,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> BuildDatasetResult:
    ann_path = project_root / "annotations" / "manual_annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise RuntimeError("manual_annotations.json format invalid: root should be dict")

    records: list[dict] = []
    skipped = 0

    image_root = project_root / "images" / "cam1"
    for image_name, payload in raw.items():
        image_path = image_root / image_name
        if not image_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        h, w = image.shape[:2]

        if isinstance(payload, list):
            human_points_raw = []
            archery_points_raw = payload
        elif isinstance(payload, dict):
            human_points_raw = payload.get("human_keypoints", [])
            archery_points_raw = payload.get("archery_keypoints", [])
        else:
            skipped += 1
            continue

        human_points = _order_points(human_points_raw, HALPE26_NAMES)
        archery_points = _order_points(archery_points_raw, ARCHERY_KEYPOINTS)

        records.append(
            {
                "image_name": image_name,
                "image_path": str(Path("images") / "cam1" / image_name).replace("\\", "/"),
                "width": int(w),
                "height": int(h),
                "human_keypoints": human_points,
                "archery_keypoints": archery_points,
            }
        )

    if not records:
        raise RuntimeError("No valid records found to build dataset")

    rng = random.Random(random_seed)
    rng.shuffle(records)

    ratio = max(0.0, min(0.9, float(val_ratio)))
    val_count = int(round(len(records) * ratio))
    if len(records) > 1:
        val_count = max(1, min(len(records) - 1, val_count))
    else:
        val_count = 0

    val_records = records[:val_count]
    train_records = records[val_count:]

    output_dir = project_root / "output" / "dataset_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_dir / "train.jsonl", train_records)
    _write_jsonl(output_dir / "val.jsonl", val_records)

    summary = {
        "total_records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "skipped_records": skipped,
        "val_ratio": ratio,
        "random_seed": random_seed,
        "format": {
            "human_keypoints_order": HALPE26_NAMES,
            "archery_keypoints_order": ARCHERY_KEYPOINTS,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return BuildDatasetResult(
        total_records=len(records),
        train_records=len(train_records),
        val_records=len(val_records),
        skipped_records=skipped,
        output_dir=output_dir,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _order_points(raw_points: object, names_order: list[str]) -> list[dict[str, float | int | str]]:
    points = raw_points if isinstance(raw_points, list) else []
    by_name: dict[str, dict] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        name = str(point.get("name", "")).strip()
        if not name:
            continue
        by_name[name] = point

    ordered: list[dict[str, float | int | str]] = []
    for name in names_order:
        point = by_name.get(name)
        if point is None:
            ordered.append(
                {
                    "name": name,
                    "x": 0.0,
                    "y": 0.0,
                    "score": 0.0,
                    "v": 0,
                }
            )
            continue

        ordered.append(
            {
                "name": name,
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "score": float(point.get("score", 0.0)),
                "v": int(point.get("v", 2)),
            }
        )

    return ordered
