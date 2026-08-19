"""Annotated MP4 output. Draws the pose skeleton over the source frames.

M1 scope: live skeleton only. The ghosted-address skeleton, butt line, and
spine-angle overlay (PLAN.md section 9) need segment.py and metrics.py to
know what "address" is, and land in M4.
"""

from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from swingcoach.pose import Landmarks

VISIBILITY_THRESHOLD = 0.3

_CONNECTIONS = sorted({(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS})

LIVE_COLOR = (0, 255, 0)  # green, BGR
LOW_CONF_COLOR = (0, 165, 255)  # orange — visible but below threshold


def _draw_frame(frame: np.ndarray, xy: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """xy: (33, 2) pixels. visibility: (33,)."""
    for a, b in _CONNECTIONS:
        if visibility[a] < VISIBILITY_THRESHOLD or visibility[b] < VISIBILITY_THRESHOLD:
            continue
        if np.isnan(xy[a]).any() or np.isnan(xy[b]).any():
            continue
        pa = tuple(xy[a].astype(int))
        pb = tuple(xy[b].astype(int))
        cv2.line(frame, pa, pb, LIVE_COLOR, 2)

    for j in range(xy.shape[0]):
        if np.isnan(xy[j]).any():
            continue
        color = LIVE_COLOR if visibility[j] >= VISIBILITY_THRESHOLD else LOW_CONF_COLOR
        cv2.circle(frame, tuple(xy[j].astype(int)), 4, color, -1)

    return frame


def render_skeleton_overlay(
    frames: list[np.ndarray],
    landmarks: Landmarks,
    out_path: Path,
) -> None:
    """Write `frames` with the live skeleton burned in to `out_path` as MP4."""
    if len(frames) != landmarks.n_frames:
        raise ValueError(
            f"frame count mismatch: {len(frames)} frames vs {landmarks.n_frames} landmark rows"
        )

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, landmarks.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    try:
        for i, frame in enumerate(frames):
            annotated = _draw_frame(frame.copy(), landmarks.xy[i], landmarks.visibility[i])
            cv2.putText(
                annotated,
                f"frame {i}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            writer.write(annotated)
    finally:
        writer.release()
