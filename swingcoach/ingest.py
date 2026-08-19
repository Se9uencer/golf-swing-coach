"""Video file -> frames + fps. The only thing this module does is read a file."""

from pathlib import Path

import cv2
import numpy as np


class VideoLoadError(RuntimeError):
    pass


def load_video(path: Path) -> tuple[list[np.ndarray], float]:
    """Decode every frame of `path` and return (frames, fps).

    Frames are BGR (OpenCV convention), in decode order.

    `fps` is read from the container header. Treat it as provisional: iPhone
    slo-mo exports can report a header fps that does not match the true
    capture rate (see MISTAKES.md, "iPhone slo-mo files lie about their frame
    rate"). This function does not attempt to correct it — that requires
    knowing expected swing timing, which is segment.py's job, not ingest's.
    Callers driving segmentation should cross-check `fps` against observed
    swing duration before trusting it.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoLoadError(f"could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise VideoLoadError(f"{path}: container reports no usable fps ({fps!r})")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise VideoLoadError(f"{path}: decoded zero frames")

    return frames, float(fps)
