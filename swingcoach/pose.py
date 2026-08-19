"""Frames -> landmark array. The only ML model call in the pipeline."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

N_LANDMARKS = 33

# MediaPipe's fixed 33-point topology. Only the subset PLAN.md's metrics
# actually use gets a name here; the rest stay index-only in `xy`/`visibility`.
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"


class PoseModelMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Landmarks:
    """Pose over a whole clip. Pixel space, OpenCV convention (y down)."""

    xy: np.ndarray  # (n_frames, 33, 2) float32, pixels
    visibility: np.ndarray  # (n_frames, 33) float32, 0..1
    fps: float

    @property
    def n_frames(self) -> int:
        return self.xy.shape[0]


def run_pose(
    frames: list[np.ndarray],
    fps: float,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Landmarks:
    """Run pose landmark detection over every frame of a clip.

    Frames with no detected pose get NaN in `xy` and 0.0 in `visibility` for
    every landmark — never a fabricated position (AGENTS.md: never fabricate
    a measurement to fill a slot). Downstream code (smooth.py) is expected to
    interpolate across these gaps explicitly, not silently.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise PoseModelMissing(
            f"pose model not found at {model_path}. See models/README.md for the download command."
        )

    h, w = frames[0].shape[:2]
    n = len(frames)
    xy = np.full((n, N_LANDMARKS, 2), np.nan, dtype=np.float32)
    visibility = np.zeros((n, N_LANDMARKS), dtype=np.float32)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame_bgr in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(i * (1000.0 / fps))

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]
            for j, lm in enumerate(landmarks):
                # Normalized [0,1] coords are not square with the frame —
                # scale x and y by width and height separately. Getting this
                # wrong silently distorts every downstream angle. See
                # AGENTS.md "MediaPipe normalized coordinates are not square".
                xy[i, j, 0] = lm.x * w
                xy[i, j, 1] = lm.y * h
                visibility[i, j] = lm.visibility

    return Landmarks(xy=xy, visibility=visibility, fps=fps)
