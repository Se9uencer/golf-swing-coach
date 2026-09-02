"""Unit tests for job-directory expiry. `sweep_expired` is what both the
web app's background cleanup thread and (indirectly) the retention promise
in README.md/PLAN.md §10 rest on."""

import os
from pathlib import Path

from swingcoach.web.storage import sweep_expired

HOUR = 3600.0


def _touch(path: Path, mtime: float, *, is_dir: bool) -> None:
    if is_dir:
        path.mkdir(parents=True)
    else:
        path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_deletes_only_expired(tmp_path):
    now = 1_000_000.0
    old = tmp_path / "old_job"
    fresh = tmp_path / "fresh_job"
    _touch(old, now - 49 * HOUR, is_dir=True)
    _touch(fresh, now - 1 * HOUR, is_dir=True)

    deleted = sweep_expired(tmp_path, expiry_hours=48.0, now=now)

    assert deleted == ["old_job"]
    assert not old.exists()
    assert fresh.exists()


def test_ignores_non_directories(tmp_path):
    now = 1_000_000.0
    stray_file = tmp_path / "not_a_job.txt"
    _touch(stray_file, now - 100 * HOUR, is_dir=False)

    deleted = sweep_expired(tmp_path, expiry_hours=48.0, now=now)

    assert deleted == []
    assert stray_file.exists()


def test_missing_storage_dir_returns_empty(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert sweep_expired(missing, expiry_hours=48.0) == []
