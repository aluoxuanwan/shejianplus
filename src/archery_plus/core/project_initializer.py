from __future__ import annotations

from pathlib import Path

PROJECT_SUBDIRS = [
    "images/cam1",
    "images/cam2",
    "annotations",
    "models",
    "calibration",
    "output",
]


def initialize_project_structure(project_root: Path) -> list[Path]:
    created_dirs: list[Path] = []
    for subdir in PROJECT_SUBDIRS:
        full_path = project_root / subdir
        full_path.mkdir(parents=True, exist_ok=True)
        created_dirs.append(full_path)
    return created_dirs
