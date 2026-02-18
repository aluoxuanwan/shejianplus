from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEYPOINT_SCHEMA_VERSION = 1
KEYPOINT_SET_DIR = Path("config") / "keypoint_sets"
KEYPOINT_SET_INDEX = "project_keypoint_sets.json"

TARGET_HUMAN = "human"
TARGET_ARCHERY = "archery"
TARGETS = (TARGET_HUMAN, TARGET_ARCHERY)

DEFAULT_HUMAN_KEYPOINT_NAMES = [
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
    "Head",
    "Neck",
    "Hip",
    "LBigToe",
    "RBigToe",
    "LSmallToe",
    "RSmallToe",
    "LHeel",
    "RHeel",
]

DEFAULT_ARCHERY_KEYPOINT_NAMES = ["UP", "DOWN", "FL", "ST", "FS"]


class KeypointSchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeypointSet:
    target: str
    set_name: str
    keypoints: list[dict[str, Any]]
    file_path: Path

    @property
    def names(self) -> list[str]:
        return [str(item["name"]) for item in self.keypoints]

    @property
    def count(self) -> int:
        return len(self.keypoints)


def normalize_target_name(raw: str) -> str:
    text = (raw or "").strip().lower()
    if text in {TARGET_HUMAN, "human_keypoints", "人体关键点"}:
        return TARGET_HUMAN
    if text in {TARGET_ARCHERY, "archery_keypoints", "弓箭关键点"}:
        return TARGET_ARCHERY
    raise KeypointSchemaError(f"Unsupported target: {raw}")


def ensure_project_keypoint_sets(project_root: Path) -> None:
    keypoint_dir = project_root / KEYPOINT_SET_DIR
    keypoint_dir.mkdir(parents=True, exist_ok=True)

    defaults = {
        TARGET_HUMAN: _build_default_payload(
            set_name="human_halpe26_default",
            keypoint_names=DEFAULT_HUMAN_KEYPOINT_NAMES,
        ),
        TARGET_ARCHERY: _build_default_payload(
            set_name="archery_5_default",
            keypoint_names=DEFAULT_ARCHERY_KEYPOINT_NAMES,
        ),
    }

    default_files = {
        TARGET_HUMAN: "human_default_halpe26.json",
        TARGET_ARCHERY: "archery_default_5.json",
    }

    for target, file_name in default_files.items():
        file_path = keypoint_dir / file_name
        if not file_path.exists():
            _write_payload(file_path, defaults[target])

    index = _read_index(project_root)
    changed = False
    for target in TARGETS:
        active_name = str(index["active"].get(target, "")).strip()
        if not active_name:
            index["active"][target] = default_files[target]
            changed = True
            continue

        active_path = keypoint_dir / active_name
        if not active_path.exists():
            index["active"][target] = default_files[target]
            changed = True
            continue

        try:
            _ = load_keypoint_set_from_file(active_path, target)
        except Exception:
            index["active"][target] = default_files[target]
            changed = True

    if changed:
        _write_index(project_root, index)


def get_active_keypoint_set(project_root: Path, target: str) -> KeypointSet:
    normalized_target = normalize_target_name(target)
    ensure_project_keypoint_sets(project_root)

    index = _read_index(project_root)
    keypoint_dir = project_root / KEYPOINT_SET_DIR
    file_name = str(index["active"].get(normalized_target, "")).strip()
    if not file_name:
        raise KeypointSchemaError(f"No active keypoint set for target: {normalized_target}")

    file_path = keypoint_dir / file_name
    if not file_path.exists():
        raise KeypointSchemaError(f"Active keypoint set file missing: {file_path}")

    return load_keypoint_set_from_file(file_path, normalized_target)


def import_keypoint_set_for_target(project_root: Path, source_path: Path, target: str) -> KeypointSet:
    normalized_target = normalize_target_name(target)
    ensure_project_keypoint_sets(project_root)

    if not source_path.exists():
        raise KeypointSchemaError(f"Keypoint set file not found: {source_path}")
    if source_path.suffix.lower() != ".json":
        raise KeypointSchemaError("Only JSON keypoint set files are supported.")

    payload = _load_and_validate_payload(source_path)
    keypoint_dir = project_root / KEYPOINT_SET_DIR
    keypoint_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(payload["set_name"])
    file_name = _next_available_filename(keypoint_dir, prefix=f"{normalized_target}_{slug}", suffix=".json")
    destination = keypoint_dir / file_name
    _write_payload(destination, payload)

    index = _read_index(project_root)
    index["active"][normalized_target] = file_name
    _write_index(project_root, index)

    return load_keypoint_set_from_file(destination, normalized_target)


def load_keypoint_set_from_file(path: Path, target: str) -> KeypointSet:
    normalized_target = normalize_target_name(target)
    payload = _load_and_validate_payload(path)
    keypoints = list(payload["keypoints"])
    return KeypointSet(
        target=normalized_target,
        set_name=str(payload["set_name"]),
        keypoints=keypoints,
        file_path=path,
    )


def _read_index(project_root: Path) -> dict[str, Any]:
    keypoint_dir = project_root / KEYPOINT_SET_DIR
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    index_path = keypoint_dir / KEYPOINT_SET_INDEX
    if not index_path.exists():
        index = {
            "schema_version": KEYPOINT_SCHEMA_VERSION,
            "active": {
                TARGET_HUMAN: "human_default_halpe26.json",
                TARGET_ARCHERY: "archery_default_5.json",
            },
        }
        _write_index(project_root, index)
        return index

    try:
        with index_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        raise KeypointSchemaError(f"Failed to read keypoint index: {index_path} | {exc}") from exc

    if not isinstance(raw, dict):
        raise KeypointSchemaError(f"Invalid keypoint index format: {index_path}")

    active = raw.get("active")
    if not isinstance(active, dict):
        active = {}
    normalized = {
        "schema_version": KEYPOINT_SCHEMA_VERSION,
        "active": {
            TARGET_HUMAN: str(active.get(TARGET_HUMAN, "")).strip(),
            TARGET_ARCHERY: str(active.get(TARGET_ARCHERY, "")).strip(),
        },
    }
    return normalized


def _write_index(project_root: Path, payload: dict[str, Any]) -> None:
    keypoint_dir = project_root / KEYPOINT_SET_DIR
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    index_path = keypoint_dir / KEYPOINT_SET_INDEX
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_default_payload(set_name: str, keypoint_names: list[str]) -> dict[str, Any]:
    return {
        "schema_version": KEYPOINT_SCHEMA_VERSION,
        "set_name": set_name,
        "keypoints": [{"id": idx, "name": name} for idx, name in enumerate(keypoint_names)],
    }


def _load_and_validate_payload(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        raise KeypointSchemaError(f"Invalid JSON file: {path} | {exc}") from exc

    if not isinstance(raw, dict):
        raise KeypointSchemaError("Keypoint set root must be JSON object.")

    schema_version = int(raw.get("schema_version", -1))
    if schema_version != KEYPOINT_SCHEMA_VERSION:
        raise KeypointSchemaError(
            f"Unsupported schema_version: {schema_version}, expected: {KEYPOINT_SCHEMA_VERSION}"
        )

    set_name = str(raw.get("set_name", "")).strip()
    if not set_name:
        raise KeypointSchemaError("Field 'set_name' is required.")

    raw_keypoints = raw.get("keypoints")
    if not isinstance(raw_keypoints, list) or not raw_keypoints:
        raise KeypointSchemaError("Field 'keypoints' must be a non-empty array.")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()

    for idx, item in enumerate(raw_keypoints):
        if not isinstance(item, dict):
            raise KeypointSchemaError(f"keypoints[{idx}] must be object.")

        if "id" not in item:
            raise KeypointSchemaError(f"keypoints[{idx}] missing 'id'.")
        if "name" not in item:
            raise KeypointSchemaError(f"keypoints[{idx}] missing 'name'.")

        keypoint_id = int(item["id"])
        keypoint_name = str(item["name"]).strip()
        if keypoint_id < 0:
            raise KeypointSchemaError(f"keypoints[{idx}] id must be >= 0.")
        if not keypoint_name:
            raise KeypointSchemaError(f"keypoints[{idx}] name cannot be empty.")
        if keypoint_id in seen_ids:
            raise KeypointSchemaError(f"Duplicate keypoint id: {keypoint_id}")
        if keypoint_name in seen_names:
            raise KeypointSchemaError(f"Duplicate keypoint name: {keypoint_name}")

        seen_ids.add(keypoint_id)
        seen_names.add(keypoint_name)
        normalized.append({"id": keypoint_id, "name": keypoint_name})

    normalized.sort(key=lambda x: int(x["id"]))
    expected_ids = list(range(len(normalized)))
    actual_ids = [int(item["id"]) for item in normalized]
    if actual_ids != expected_ids:
        raise KeypointSchemaError(
            f"Keypoint ids must be continuous 0..N-1, got: {actual_ids}"
        )

    return {
        "schema_version": KEYPOINT_SCHEMA_VERSION,
        "set_name": set_name,
        "keypoints": normalized,
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _slugify(text: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip())
    clean = re.sub(r"_+", "_", clean).strip("_").lower()
    return clean or "set"


def _next_available_filename(base_dir: Path, prefix: str, suffix: str) -> str:
    candidate = f"{prefix}{suffix}"
    if not (base_dir / candidate).exists():
        return candidate

    index = 1
    while True:
        candidate = f"{prefix}_{index}{suffix}"
        if not (base_dir / candidate).exists():
            return candidate
        index += 1
