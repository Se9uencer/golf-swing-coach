"""Environment-driven settings for the public web front end.

Everything here has a sane default so the app runs locally with zero
configuration; Render (see render.yaml) overrides STORAGE_DIR to point at
the mounted persistent disk.
"""

import os
from pathlib import Path

from swingcoach.pose import DEFAULT_MODEL_PATH


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


STORAGE_DIR = Path(os.environ.get("SWINGCOACH_STORAGE_DIR", "./data")).resolve()
MODEL_PATH = Path(os.environ.get("SWINGCOACH_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# Upload caps -- the main lever against a public, anonymous tool being used
# to run unbounded CPU-bound pose estimation for free. Pose estimation is
# ~10fps on CPU, so these bound worst-case processing time per job as well
# as abuse cost, not just upload bandwidth.
MAX_UPLOAD_BYTES = _env_int("SWINGCOACH_MAX_UPLOAD_BYTES", 300 * 1024 * 1024)  # 300 MB
MAX_DURATION_S = _env_float("SWINGCOACH_MAX_DURATION_S", 60.0)

# How long a finished job's video + report stay on disk before the cleanup
# sweep deletes them. Reports are reachable only via their unguessable
# job-id URL -- nothing lists or indexes them.
EXPIRY_HOURS = _env_float("SWINGCOACH_EXPIRY_HOURS", 48.0)
CLEANUP_INTERVAL_S = _env_float("SWINGCOACH_CLEANUP_INTERVAL_S", 3600.0)

# In-process job queue. Concurrency defaults to 1: mediapipe pose estimation
# is CPU-bound, so running two jobs "concurrently" on a small instance just
# makes both slower, not faster. MAX_QUEUE_DEPTH is the blunt global safety
# valve behind per-IP rate limiting -- a botnet spread across many IPs could
# still pile up jobs, so new uploads are rejected once this many are
# queued+processing.
WORKER_CONCURRENCY = _env_int("SWINGCOACH_WORKER_CONCURRENCY", 1)
MAX_QUEUE_DEPTH = _env_int("SWINGCOACH_MAX_QUEUE_DEPTH", 10)

# Per-IP rate limiting, fixed hourly window.
RATE_LIMIT_PER_HOUR = _env_int("SWINGCOACH_RATE_LIMIT_PER_HOUR", 5)
