"""H.264 MP4 writer via a piped ffmpeg subprocess.

cv2.VideoWriter's bundled FFmpeg has no H.264 encoder available in this
environment -- confirmed directly: avc1, H264, h264, and X264 fourcc codes
all fail to open a writer at all; only mp4v works. An mp4v-encoded file is
a valid MP4, but ordinary browsers (Chrome, Safari, Firefox) only decode
H.264 or VP9/AV1 for <video>, not MPEG-4 Part 2 -- so every report opened
fine and every embedded video was silently a black box with no picture.
Confirmed by a real user downloading a real generated report. See
MISTAKES.md.

imageio-ffmpeg ships a real static ffmpeg binary with libx264 built in.
Frames are piped to it directly as raw BGR24 rather than round-tripping
through cv2.VideoWriter's mp4v first, which would be a lossy double-encode
for no reason.
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


class Mp4Writer:
    """Same call pattern as cv2.VideoWriter: write(frame) then release()."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self._proc = subprocess.Popen(
            [
                ffmpeg_exe,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",  # required for broad browser/QuickTime compatibility
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def isOpened(self) -> bool:  # noqa: N802 -- matches cv2.VideoWriter's API
        return self._proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
