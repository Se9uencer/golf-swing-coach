"""Annotated MP4 output. Draws the pose skeleton over the source frames.

Two renderers live here:
- `render_skeleton_overlay`: whole-clip, live skeleton only. The M1 debug
  view -- "does pose tracking look right on this clip at all."
- `render_swing_overlay`: one swing's frame range, with the ghosted-address
  comparison PLAN.md section 9 describes -- this is the actual coaching
  output, and what report.py embeds per swing.
"""

from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from swingcoach.metrics import head_trace
from swingcoach.pose import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmarks,
    weighted_midpoint,
)
from swingcoach.render.video_writer import Mp4Writer
from swingcoach.segment import Swing

VISIBILITY_THRESHOLD = 0.3
GHOST_ALPHA = 0.55  # how strongly the frozen address skeleton shows through

LIVE_COLOR = (0, 255, 0)  # green, BGR
LOW_CONF_COLOR = (0, 165, 255)  # orange -- visible but below threshold
# Light gray at low alpha was invisible against a bright sky in real footage
# (verified: real pixels changed, just not legibly) -- bold red-orange reads
# against sky, grass, and clothing alike.
GHOST_COLOR = (0, 80, 255)  # red-orange -- the frozen address skeleton
BUTT_LINE_COLOR = (0, 230, 255)  # yellow -- vertical line at address hip position
SPINE_LIVE_COLOR = (255, 200, 0)  # cyan-ish -- live pelvis-to-shoulder line
SPINE_GHOST_COLOR = (0, 80, 255)  # matches GHOST_COLOR -- address pelvis-to-shoulder line
HEAD_GHOST_COLOR = (200, 200, 255)  # pink-gray -- address head marker
HAND_PATH_COLOR = (255, 0, 255)  # magenta -- accumulating hand trace

_CONNECTIONS = sorted({(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS})


def _draw_skeleton(frame: np.ndarray, xy: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """The full 33-point skeleton, live-tracking color scheme."""
    for a, b in _CONNECTIONS:
        if visibility[a] < VISIBILITY_THRESHOLD or visibility[b] < VISIBILITY_THRESHOLD:
            continue
        if np.isnan(xy[a]).any() or np.isnan(xy[b]).any():
            continue
        cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), LIVE_COLOR, 2)

    for j in range(xy.shape[0]):
        if np.isnan(xy[j]).any():
            continue
        color = LIVE_COLOR if visibility[j] >= VISIBILITY_THRESHOLD else LOW_CONF_COLOR
        cv2.circle(frame, tuple(xy[j].astype(int)), 4, color, -1)

    return frame


def _draw_ghost_skeleton(canvas: np.ndarray, xy: np.ndarray, visibility: np.ndarray) -> None:
    """The address-frame skeleton, drawn onto `canvas` in a flat ghost color
    for later alpha-blending. Not confidence-gated as tightly as the live
    skeleton -- address is a single still frame, chosen for exactly this
    purpose, so a slightly lower bar for "draw it" is fine here."""
    for a, b in _CONNECTIONS:
        if visibility[a] < VISIBILITY_THRESHOLD or visibility[b] < VISIBILITY_THRESHOLD:
            continue
        if np.isnan(xy[a]).any() or np.isnan(xy[b]).any():
            continue
        cv2.line(canvas, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), GHOST_COLOR, 2)
    for j in range(xy.shape[0]):
        if np.isnan(xy[j]).any() or visibility[j] < VISIBILITY_THRESHOLD:
            continue
        cv2.circle(canvas, tuple(xy[j].astype(int)), 4, GHOST_COLOR, -1)


def render_skeleton_overlay(frames: list[np.ndarray], landmarks: Landmarks, out_path: Path) -> None:
    """Write `frames` with the live skeleton burned in to `out_path` as MP4."""
    if len(frames) != landmarks.n_frames:
        raise ValueError(
            f"frame count mismatch: {len(frames)} frames vs {landmarks.n_frames} landmark rows"
        )

    h, w = frames[0].shape[:2]
    writer = Mp4Writer(out_path, landmarks.fps, w, h)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    try:
        for i, frame in enumerate(frames):
            annotated = _draw_skeleton(frame.copy(), landmarks.xy[i], landmarks.visibility[i])
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


def _phase_for(frame_idx: int, swing: Swing) -> str:
    if frame_idx <= swing.address + 1:
        return "address"
    if frame_idx < swing.top:
        return "backswing"
    if frame_idx < swing.impact:
        return "downswing"
    if frame_idx == swing.impact:
        return "impact"
    return "follow-through"


def render_swing_overlay(
    frames: list[np.ndarray],
    landmarks: Landmarks,
    swing: Swing,
    swing_number: int,
    out_path: Path,
) -> None:
    """Write the [swing.start, swing.end] slice of `frames` to `out_path`,
    with the live skeleton plus PLAN.md section 9's ghosted-address
    comparison burned in: the address-frame skeleton frozen and shown
    translucent, a vertical "butt line" at the address hip position (early
    extension is crossing it), the address-vs-live spine line, an address
    head marker, and an accumulating hand-path trace. This is the actual
    coaching visual -- the deviation numbers in torso-lengths and degrees
    are an abstraction of exactly what this draws.
    """
    if len(frames) != landmarks.n_frames:
        raise ValueError(
            f"frame count mismatch: {len(frames)} frames vs {landmarks.n_frames} landmark rows"
        )

    h, w = frames[0].shape[:2]
    address = swing.address

    pelvis_xy, _ = weighted_midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    shoulder_xy, _ = weighted_midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    wrist_xy, _ = weighted_midpoint(landmarks, LEFT_WRIST, RIGHT_WRIST)
    head_xy, _, _ = head_trace(landmarks)

    butt_line_x = pelvis_xy[address, 0]
    ghost_spine_a = pelvis_xy[address]
    ghost_spine_b = shoulder_xy[address]
    ghost_head = head_xy[address]
    ghost_valid = not (np.isnan(ghost_head).any())

    writer = Mp4Writer(out_path, landmarks.fps, w, h)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    hand_path_points: list[tuple[int, int]] = []

    try:
        for i in range(swing.start, swing.end + 1):
            frame = frames[i].copy()

            # Ghost layer: address skeleton, spine line, head marker, drawn
            # onto a copy of the frame and alpha-blended in, so it reads as
            # "faintly present" rather than competing with the live tracking.
            ghost = frame.copy()
            if not np.isnan(butt_line_x):
                cv2.line(ghost, (int(butt_line_x), 0), (int(butt_line_x), h), BUTT_LINE_COLOR, 2)
            if not np.isnan(ghost_spine_a).any() and not np.isnan(ghost_spine_b).any():
                cv2.line(
                    ghost,
                    tuple(ghost_spine_a.astype(int)),
                    tuple(ghost_spine_b.astype(int)),
                    SPINE_GHOST_COLOR,
                    3,
                )
            _draw_ghost_skeleton(ghost, landmarks.xy[address], landmarks.visibility[address])
            if ghost_valid:
                cv2.circle(ghost, tuple(ghost_head.astype(int)), 8, HEAD_GHOST_COLOR, 2)
            frame = cv2.addWeighted(ghost, GHOST_ALPHA, frame, 1 - GHOST_ALPHA, 0)

            # Butt line redrawn crisp on top (not blended) so it stays a
            # legible reference, not just a faint suggestion.
            if not np.isnan(butt_line_x):
                cv2.line(frame, (int(butt_line_x), 0), (int(butt_line_x), h), BUTT_LINE_COLOR, 1)

            # Live layer.
            frame = _draw_skeleton(frame, landmarks.xy[i], landmarks.visibility[i])
            if not np.isnan(pelvis_xy[i]).any() and not np.isnan(shoulder_xy[i]).any():
                cv2.line(
                    frame,
                    tuple(pelvis_xy[i].astype(int)),
                    tuple(shoulder_xy[i].astype(int)),
                    SPINE_LIVE_COLOR,
                    2,
                )

            # Accumulating hand path.
            if not np.isnan(wrist_xy[i]).any():
                hand_path_points.append(tuple(wrist_xy[i].astype(int)))
            for p in range(1, len(hand_path_points)):
                cv2.line(frame, hand_path_points[p - 1], hand_path_points[p], HAND_PATH_COLOR, 2)

            phase = _phase_for(i, swing)
            cv2.putText(
                frame,
                f"swing {swing_number}  {phase}  frame {i}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            writer.write(frame)
    finally:
        writer.release()
