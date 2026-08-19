"""Diagnostic plot: wrist speed over time with detected swing events marked.

Exists because a segmentation question ("why did it miss this swing", "why
did it split this one wrong") is answered in five seconds by looking at
this plot and in much longer by staring at segment.py. See AGENTS.md, "When
a number looks wrong, plot it before theorizing."

Two entry points, one shared figure: `save_velocity_plot` writes a PNG file
(the standalone debug view); `velocity_plot_png_bytes` returns the same
image as bytes, for report.py to embed directly without a temp file.
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swingcoach.pose import Landmarks
from swingcoach.segment import Swing, wrist_speed

_EVENT_STYLE = {
    "address": ("#2b6cb0", "-."),
    "top": ("#dd6b20", "--"),
    "impact": ("#c53030", "-"),
}


def _build_figure(lm: Landmarks, swings: list[Swing], fps: float):
    speed = wrist_speed(lm, fps)
    t = np.arange(len(speed)) / fps

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, speed, color="#333333", linewidth=1, label="wrist speed (px/s)")

    for i, swing in enumerate(swings):
        ax.axvspan(swing.start / fps, swing.end / fps, color="#2b6cb0", alpha=0.06)
        for event_name, frame in (
            ("address", swing.address),
            ("top", swing.top),
            ("impact", swing.impact),
        ):
            color, style = _EVENT_STYLE[event_name]
            ax.axvline(frame / fps, color=color, linestyle=style, linewidth=1, alpha=0.8)
        ax.text(
            swing.start / fps,
            ax.get_ylim()[1] * 0.95,
            f"swing {i}",
            fontsize=8,
            color="#2b6cb0",
        )

    handles = [
        plt.Line2D([0], [0], color=color, linestyle=style, label=name)
        for name, (color, style) in _EVENT_STYLE.items()
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wrist speed (px/s)")
    ax.set_title(f"{len(swings)} swing(s) detected")
    fig.tight_layout()
    return fig


def save_velocity_plot(lm: Landmarks, swings: list[Swing], fps: float, out_path: Path) -> None:
    fig = _build_figure(lm, swings, fps)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def velocity_plot_png_bytes(lm: Landmarks, swings: list[Swing], fps: float) -> bytes:
    fig = _build_figure(lm, swings, fps)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()
