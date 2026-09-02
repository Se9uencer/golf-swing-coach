"""One job's worth of work: uploaded video in, report.html out.

Same pipeline as `swingcoach.cli`'s `report` command (ingest -> pose ->
segment -> render), just driven by a job id and storage paths instead of
argv, and raising instead of printing to stderr so the caller (jobs.py)
can attach the message to job status.
"""

from pathlib import Path

from swingcoach.ingest import VideoLoadError, load_video
from swingcoach.pose import PoseModelMissing, run_pose
from swingcoach.render.report import generate_report
from swingcoach.segment import find_swings
from swingcoach.web import config
from swingcoach.web.storage import job_dir, report_path


class NoSwingsDetected(RuntimeError):
    pass


def process(storage_dir: Path, job_id: str, source_name: str) -> Path:
    """Run the pipeline for `job_id`'s uploaded input and write its report.
    Returns the report path. Deletes the uploaded input video on success --
    the report already carries everything from it (overlay video, stills)
    that a viewer needs, so there's no reason to keep the raw upload around
    for the rest of its retention window.
    """
    matches = sorted(job_dir(storage_dir, job_id).glob("input.*"))
    if not matches:
        raise VideoLoadError(f"no uploaded input found for job {job_id}")
    src = matches[0]

    frames, fps = load_video(src)

    try:
        landmarks = run_pose(frames, fps, model_path=config.MODEL_PATH)
    except PoseModelMissing as e:
        raise RuntimeError(str(e)) from e

    swings = find_swings(landmarks, fps)
    if not swings:
        raise NoSwingsDetected(
            "No swings detected. Check your camera angle and framing -- "
            "down-the-line, full body in frame, camera braced still."
        )

    out_path = report_path(storage_dir, job_id)
    generate_report(frames, landmarks, swings, fps, source_name, out_path)

    src.unlink(missing_ok=True)
    return out_path
