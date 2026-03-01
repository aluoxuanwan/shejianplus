from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shejianplus.config import BASE_DIR, CONFIG_FILES, MODEL_FILES, WEIGHT_FILES


@dataclass(frozen=True)
class ArtifactInfo:
    key: str
    file_name: str
    purpose: str
    path: Path
    exists: bool


def _to_artifact(key: str, file_name: str, purpose: str) -> ArtifactInfo:
    path = BASE_DIR / file_name
    return ArtifactInfo(
        key=key,
        file_name=file_name,
        purpose=purpose,
        path=path,
        exists=path.exists(),
    )


def collect_artifacts() -> list[ArtifactInfo]:
    artifacts: list[ArtifactInfo] = []

    for key, file_name in MODEL_FILES.items():
        purpose = "ONNX推理模型"
        artifacts.append(_to_artifact(key, file_name, purpose))

    for key, file_name in WEIGHT_FILES.items():
        purpose = "训练权重/预训练权重"
        artifacts.append(_to_artifact(key, file_name, purpose))

    for key, file_name in CONFIG_FILES.items():
        purpose = "训练配置文件"
        artifacts.append(_to_artifact(key, file_name, purpose))

    return artifacts


def count_ready_models() -> tuple[int, int]:
    model_artifacts = [_to_artifact(k, v, "ONNX推理模型") for k, v in MODEL_FILES.items()]
    ready = sum(1 for item in model_artifacts if item.exists)
    return ready, len(model_artifacts)
