"""HTML report generation. PLAN.md section 9: one file, opens in a browser,
readable on a phone in the car after the range.

Two output shapes share the same underlying data:
- `generate_report`: a complete, self-contained HTML document (its own
  doctype/html/head/body) meant to be downloaded and opened locally.
- `generate_artifact_html`: a fragment (title + style + body content, no
  wrapping document tags) meant to be published as a Claude Artifact, which
  supplies its own document shell. See render/pdf.py and MISTAKES.md for why
  video sometimes needs to be dropped from the local-file path.

This is I/O by nature (writes HTML, and internally writes temporary
per-swing videos because ffmpeg needs a real file to encode to) --
consistent with AGENTS.md's rule that render/ modules do I/O while the
analysis core stays pure. The temporary videos never outlive this module;
only the assembled HTML, with everything base64-embedded inside it, is left
behind.
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
_ARTIFACT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report_artifact.html.jinja"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _frame_still_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("failed to encode frame as PNG")
    return _b64(buf.tobytes())


def _line_dict(line):
    return {
        "name": line.name,
        "pretty_name": line.pretty_name,
        "cue": line.cue,
        "detail": line.detail,
        "value": line.value,
        "unit": line.unit,
        "peak_frame": line.peak_frame,
        "confidence": line.confidence,
    }


def _build_context(
    frames: list[np.ndarray],
    landmarks: Landmarks,
    swings: list[Swing],
    fps: float,
    source_name: str,
    include_video: bool,
) -> dict:
    """Shared data prep for both output shapes: per-swing stills, deviations,
    feedback, and the session-wide velocity plot, all base64-encoded and
    ready to hand to either Jinja template.
    """
    swing_contexts = []

    with tempfile.TemporaryDirectory(prefix="swingcoach_report_") as tmp:
        tmp_dir = Path(tmp)

        for i, swing in enumerate(swings):
            video_b64 = None
            if include_video:
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

    return {
        "source_name": source_name,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "fps": fps,
        "n_frames": landmarks.n_frames,
        "velocity_plot_b64": _b64(velocity_plot_png_bytes(landmarks, swings, fps)),
        "swings": swing_contexts,
    }


def generate_report(
    frames: list[np.ndarray],
    landmarks: Landmarks,
    swings: list[Swing],
    fps: float,
    source_name: str,
    out_path: Path,
    include_video: bool = True,
) -> None:
    """Render every swing in `swings` -- overlay video, key-frame stills,
    ranked deviations, feedback text -- plus the session-wide velocity plot,
    into a single self-contained HTML file at `out_path`.

    `include_video=False` skips rendering and embedding the per-swing overlay
    videos entirely -- used for the PDF export path (render/pdf.py), since a
    PDF can't play an embedded <video> (confirmed: it renders as a dead black
    box) and rendering/encoding video nobody will see is wasted time. The
    three key-frame stills already carry the same story for that path.
    """
    context = _build_context(frames, landmarks, swings, fps, source_name, include_video)
    template = Template(_TEMPLATE_PATH.read_text())
    html = template.render(**context)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def generate_artifact_html(
    frames: list[np.ndarray],
    landmarks: Landmarks,
    swings: list[Swing],
    fps: float,
    source_name: str,
    out_path: Path,
) -> None:
    """Same report, rendered as an HTML fragment (no doctype/html/head/body)
    for publishing via the Artifact tool, which supplies its own document
    shell and expects page content directly. Always includes video -- the
    Artifact viewer plays embedded <video> fine; only PDF export needs it
    dropped.
    """
    context = _build_context(frames, landmarks, swings, fps, source_name, include_video=True)
    template = Template(_ARTIFACT_TEMPLATE_PATH.read_text())
    html = template.render(**context)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
