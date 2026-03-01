from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class FrameExtractionResult:
    total_frames: int
    saved_frames: int
    output_dir: Path


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 5,
    image_ext: str = "jpg",
) -> FrameExtractionResult:
    if frame_interval < 1:
        raise ValueError("frame_interval must be >= 1")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = 0
    saved_frames = 0
    stem = video_path.stem

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if total_frames % frame_interval == 0:
                file_name = f"{stem}_{total_frames:06d}.{image_ext}"
                save_path = output_dir / file_name
                ok = cv2.imwrite(str(save_path), frame)
                if not ok:
                    raise RuntimeError(f"Failed to write frame: {save_path}")
                saved_frames += 1

            total_frames += 1
    finally:
        cap.release()

    return FrameExtractionResult(
        total_frames=total_frames,
        saved_frames=saved_frames,
        output_dir=output_dir,
    )
