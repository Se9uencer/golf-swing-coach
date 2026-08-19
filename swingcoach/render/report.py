"""Self-contained HTML report. PLAN.md section 9: one file, opens in a
browser, readable on a phone in the car after the range.

This is I/O by nature (writes an HTML file, and internally writes temporary
per-swing videos because OpenCV's VideoWriter needs a real file to encode
to) -- consistent with AGENTS.md's rule that render/ modules do I/O while
the analysis core stays pure. The temporary videos never outlive this
function; only the assembled HTML, with everything base64-embedded inside
it, is left behind.
"""

import base64
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from jinja2 import Template

from swingcoach.feedback import describe
from swingcoach.metrics import measure
from swingcoach.pose import Landmarks
from swingcoach.render.overlay import render_swing_overlay
from swingcoach.render.velocity_plot import velocity_plot_png_bytes
from swingcoach.segment import Swing

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html.jinja"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _frame_still_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("failed to encode frame as PNG")
    return _b64(buf.tobytes())


def _line_dict(line):
    return {
        "pretty_name": line.pretty_name,
        "text": line.text,
        "value": line.value,
        "unit": line.unit,
        "peak_frame": line.peak_frame,
        "confidence": line.confidence,
    }


def generate_report(
    frames: list[np.ndarray],
    landmarks: Landmarks,
    swings: list[Swing],
    fps: float,
    source_name: str,
    out_path: Path,
) -> None:
    """Render every swing in `swings` -- overlay video, key-frame stills,
    ranked deviations, feedback text -- plus the session-wide velocity plot,
    into a single HTML file at `out_path`.
    """
    swing_contexts = []

    with tempfile.TemporaryDirectory(prefix="swingcoach_report_") as tmp:
        tmp_dir = Path(tmp)

        for i, swing in enumerate(swings):
            video_path = tmp_dir / f"swing_{i}.mp4"
            render_swing_overlay(frames, landmarks, swing, i, video_path)
            video_b64 = _b64(video_path.read_bytes())

            deviations = measure(landmarks, swing, fps)
            fb = describe(deviations)

            swing_contexts.append(
                {
                    "index": i,
                    "video_b64": video_b64,
                    "address_frame_b64": _frame_still_b64(frames[swing.address]),
                    "top_frame_b64": _frame_still_b64(frames[swing.top]),
                    "impact_frame_b64": _frame_still_b64(frames[swing.impact]),
                    "address_idx": swing.address,
                    "top_idx": swing.top,
                    "impact_idx": swing.impact,
                    "start_s": swing.start / fps,
                    "end_s": swing.end / fps,
                    "biggest": _line_dict(fb.biggest) if fb.biggest else None,
                    "biggest_drill": fb.biggest_drill,
                    "steadiest": _line_dict(fb.steadiest) if fb.steadiest else None,
                    "lines": [_line_dict(line) for line in fb.lines],
                }
            )

    velocity_plot_b64 = _b64(velocity_plot_png_bytes(landmarks, swings, fps))

    template = Template(_TEMPLATE_PATH.read_text())
    html = template.render(
        source_name=source_name,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        fps=fps,
        n_frames=landmarks.n_frames,
        velocity_plot_b64=velocity_plot_b64,
        swings=swing_contexts,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
