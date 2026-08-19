"""Entry points for the pipeline.

    python -m swingcoach.cli overlay <input> <output> [--model PATH] [--cutoff-hz N]
    python -m swingcoach.cli segment <input> [--model PATH] [--plot PATH] [--cache PATH]

This is the only module that touches argv or prints to the user — everything
it calls is pure or does exactly the I/O its name says (AGENTS.md).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from swingcoach.ingest import VideoLoadError, load_video
from swingcoach.pose import DEFAULT_MODEL_PATH, Landmarks, PoseModelMissing, run_pose
from swingcoach.render.overlay import render_skeleton_overlay
from swingcoach.render.report import generate_report
from swingcoach.render.velocity_plot import save_velocity_plot
from swingcoach.segment import find_swings
from swingcoach.smooth import lowpass


def _load_or_run_pose(
    input_path: Path, fps: float, frames: list, model_path: Path, cache_path: Path | None
) -> Landmarks:
    """Pose is the slow step (~10fps on CPU). `--cache` skips recomputing it
    across runs while tuning segment.py's thresholds against the same clip."""
    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        print(f"loaded pose from cache {cache_path}")
        return Landmarks(xy=data["xy"], visibility=data["visibility"], fps=float(data["fps"]))

    t0 = time.time()
    landmarks = run_pose(frames, fps, model_path=model_path)
    elapsed = time.time() - t0
    detected = int((landmarks.visibility.max(axis=1) > 0).sum())
    print(
        f"pose: {detected}/{landmarks.n_frames} frames detected, "
        f"{elapsed:.1f}s ({landmarks.n_frames / elapsed:.1f} fps)"
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, xy=landmarks.xy, visibility=landmarks.visibility, fps=landmarks.fps)
        print(f"cached pose to {cache_path}")

    return landmarks


def _run_overlay(args: argparse.Namespace) -> int:
    try:
        frames, fps = load_video(args.input)
    except VideoLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{args.input}: {len(frames)} frames @ {fps:.2f}fps (container header)")

    try:
        landmarks = _load_or_run_pose(args.input, fps, frames, args.model, args.cache)
    except PoseModelMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    smoothed_xy = lowpass(landmarks.xy, fps=fps, cutoff_hz=args.cutoff_hz)
    landmarks = Landmarks(xy=smoothed_xy, visibility=landmarks.visibility, fps=fps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_skeleton_overlay(frames, landmarks, args.output)
    print(f"wrote {args.output}")
    return 0


def _run_segment(args: argparse.Namespace) -> int:
    try:
        frames, fps = load_video(args.input)
    except VideoLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{args.input}: {len(frames)} frames @ {fps:.2f}fps (container header)")

    try:
        landmarks = _load_or_run_pose(args.input, fps, frames, args.model, args.cache)
    except PoseModelMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    swings = find_swings(landmarks, fps)
    print(f"{len(swings)} swing(s) detected")
    for i, s in enumerate(swings):
        print(
            f"  swing {i}: start={s.start} address={s.address} top={s.top} "
            f"impact={s.impact} end={s.end}  "
            f"({s.start / fps:.2f}s - {s.end / fps:.2f}s)"
        )

    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        save_velocity_plot(landmarks, swings, fps, args.plot)
        print(f"wrote {args.plot}")

    return 0


def _run_report(args: argparse.Namespace) -> int:
    try:
        frames, fps = load_video(args.input)
    except VideoLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{args.input}: {len(frames)} frames @ {fps:.2f}fps (container header)")

    try:
        landmarks = _load_or_run_pose(args.input, fps, frames, args.model, args.cache)
    except PoseModelMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    swings = find_swings(landmarks, fps)
    print(f"{len(swings)} swing(s) detected")
    if not swings:
        print("nothing to report -- no swings detected", file=sys.stderr)
        return 1

    generate_report(frames, landmarks, swings, fps, args.input.name, args.output)
    print(f"wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_overlay = sub.add_parser("overlay", help="video in, skeleton-overlay video out")
    p_overlay.add_argument("input", type=Path, help="source swing video")
    p_overlay.add_argument("output", type=Path, help="annotated MP4 to write")
    p_overlay.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_overlay.add_argument(
        "--cutoff-hz",
        type=float,
        default=12.0,
        help="low-pass cutoff for skeleton smoothing (default 12Hz, see PLAN.md)",
    )
    p_overlay.add_argument("--cache", type=Path, default=None, help="pose cache .npz path")
    p_overlay.set_defaults(func=_run_overlay)

    p_segment = sub.add_parser("segment", help="split a clip into swings; print events")
    p_segment.add_argument("input", type=Path, help="source swing video")
    p_segment.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_segment.add_argument(
        "--plot", type=Path, default=None, help="write the wrist-speed diagnostic plot here"
    )
    p_segment.add_argument("--cache", type=Path, default=None, help="pose cache .npz path")
    p_segment.set_defaults(func=_run_segment)

    p_report = sub.add_parser("report", help="the actual thing: one self-contained HTML report")
    p_report.add_argument("input", type=Path, help="source swing video")
    p_report.add_argument("output", type=Path, help="HTML report to write")
    p_report.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p_report.add_argument("--cache", type=Path, default=None, help="pose cache .npz path")
    p_report.set_defaults(func=_run_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
