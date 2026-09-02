"""Filesystem layout for uploaded videos and generated reports.

One directory per job, named by its job id (an unguessable uuid4 hex --
that unguessability is the report's only access control, see PLAN.md
section 10). Age is tracked by directory mtime, not a separate database,
so `sweep_expired` and the job registry can't disagree about it.
"""

import shutil
import time
from pathlib import Path


def job_dir(storage_dir: Path, job_id: str) -> Path:
    return storage_dir / job_id


def input_path(storage_dir: Path, job_id: str, suffix: str) -> Path:
    return job_dir(storage_dir, job_id) / f"input{suffix}"


def report_path(storage_dir: Path, job_id: str) -> Path:
    return job_dir(storage_dir, job_id) / "report.html"


def sweep_expired(storage_dir: Path, expiry_hours: float, now: float | None = None) -> list[str]:
    """Delete every job directory older than `expiry_hours`. Returns the
    deleted job ids. Age is the directory's own mtime, touched by
    `job_dir(...).mkdir()` at upload time -- not the mtime of files inside
    it, which processing keeps rewriting.
    """
    if not storage_dir.exists():
        return []

    now = time.time() if now is None else now
    cutoff = expiry_hours * 3600
    deleted = []
    for entry in storage_dir.iterdir():
        if not entry.is_dir():
            continue
        if now - entry.stat().st_mtime > cutoff:
            shutil.rmtree(entry, ignore_errors=True)
            deleted.append(entry.name)
    return deleted
