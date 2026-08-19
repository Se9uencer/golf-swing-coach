"""Entry point for the M1 pipeline: video in, skeleton-overlay video out.

    python -m swingcoach.cli <input_video> <output_video> [--model PATH] [--cutoff-hz N]

This is the only module that touches argv or prints to the user — everything
it calls is pure or does exactly the I/O its name says (AGENTS.md).
"""

import argparse
import sys
import time
from pathlib import Path

from swingcoach.ingest import VideoLoadError, load_video
from swingcoach.pose import DEFAULT_MODEL_PATH, PoseModelMissing, run_pose
from swingcoach.render.overlay import render_skeleton_overlay
from swingcoach.smooth import lowpass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source swing video")
    parser.add_argument("output", type=Path, help="annotated MP4 to write")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=12.0,
        help="low-pass cutoff for skeleton smoothing (default 12Hz, see PLAN.md)",
    )
    args = parser.parse_args(argv)

    try:
        frames, fps = load_video(args.input)
    except VideoLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{args.input}: {len(frames)} frames @ {fps:.2f}fps (container header)")

    t0 = time.time()
    try:
        landmarks = run_pose(frames, fps, model_path=args.model)
    except PoseModelMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0
    detected = int((landmarks.visibility.max(axis=1) > 0).sum())
    print(
        f"pose: {detected}/{landmarks.n_frames} frames detected, "
        f"{elapsed:.1f}s ({landmarks.n_frames / elapsed:.1f} fps)"
    )

    smoothed_xy = lowpass(landmarks.xy, fps=fps, cutoff_hz=args.cutoff_hz)
    landmarks = type(landmarks)(xy=smoothed_xy, visibility=landmarks.visibility, fps=fps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_skeleton_overlay(frames, landmarks, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
