"""The public web front end. PLAN.md section 10.

Upload a swing video, get back a link to an HTML report once the pipeline
(unchanged from the CLI -- see swingcoach/web/pipeline.py) has run on it.
No accounts, no database: job status lives in memory (swingcoach.web.jobs),
and a report is reachable only by its unguessable job-id URL until it
expires and the cleanup sweep deletes it.
"""

import logging
import re
import shutil
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from swingcoach.web import config
from swingcoach.web.jobs import QueueFull, make_job_dir, new_job_id, queue_instance
from swingcoach.web.ratelimit import RateLimiter
from swingcoach.web.storage import report_path, sweep_expired

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
ALLOWED_SUFFIXES = {".mov", ".mp4", ".m4v"}
_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")  # matches uuid4().hex from jobs.new_job_id


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    queue_instance.start()
    threading.Thread(target=_cleanup_loop, daemon=True, name="swingcoach-cleanup").start()
    yield


app = FastAPI(title="swingcoach", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")

rate_limiter = RateLimiter(limit=config.RATE_LIMIT_PER_HOUR, window_s=3600.0)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _cleanup_loop() -> None:
    while True:
        time.sleep(config.CLEANUP_INTERVAL_S)
        try:
            deleted = sweep_expired(config.STORAGE_DIR, config.EXPIRY_HOURS)
            if deleted:
                logger.info("cleanup: expired %d job(s)", len(deleted))
        except Exception:
            logger.exception("cleanup sweep failed")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_duration_s": int(config.MAX_DURATION_S),
            "rate_limit_per_hour": config.RATE_LIMIT_PER_HOUR,
            "expiry_hours": int(config.EXPIRY_HOURS),
        },
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/upload")
async def upload(request: Request, video: UploadFile):
    ip = _client_ip(request)
    if not rate_limiter.allow(ip):
        message = f"Rate limit reached ({config.RATE_LIMIT_PER_HOUR}/hour). Try again later."
        return templates.TemplateResponse(
            request, "error.html", {"message": message}, status_code=429
        )

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return templates.TemplateResponse(
            request,
            "error.html",
            {"message": f"Unsupported file type {suffix!r}. Upload a .mov or .mp4 clip."},
            status_code=400,
        )

    job_id = new_job_id()
    dest_dir = make_job_dir(config.STORAGE_DIR, job_id)
    dest = dest_dir / f"input{suffix}"

    written = 0
    chunk_size = 1024 * 1024
    with dest.open("wb") as f:
        while chunk := await video.read(chunk_size):
            written += len(chunk)
            if written > config.MAX_UPLOAD_BYTES:
                f.close()
                _discard(dest_dir)
                max_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
                return templates.TemplateResponse(
                    request,
                    "error.html",
                    {"message": f"File too large. Max {max_mb}MB."},
                    status_code=400,
                )
            f.write(chunk)

    duration_error = _reject_if_invalid_or_too_long(dest)
    if duration_error:
        _discard(dest_dir)
        return templates.TemplateResponse(
            request, "error.html", {"message": duration_error}, status_code=400
        )

    try:
        queue_instance.submit(job_id, video.filename or "swing")
    except QueueFull as e:
        _discard(dest_dir)
        return templates.TemplateResponse(
            request, "error.html", {"message": str(e)}, status_code=503
        )

    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


def _discard(dest_dir: Path) -> None:
    shutil.rmtree(dest_dir, ignore_errors=True)


def _reject_if_invalid_or_too_long(path: Path) -> str | None:
    """Cheap metadata-only probe (no full frame decode) so obviously bad or
    over-length uploads are rejected before they take a queue slot. The
    real pipeline still does its own full validation -- this is a fast
    pre-filter, not a replacement for it.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return "Could not read this as a video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if not fps or fps <= 0:
        return "This video's container doesn't report a usable frame rate."
    if frame_count and frame_count / fps > config.MAX_DURATION_S:
        return f"Clip too long. Max {int(config.MAX_DURATION_S)}s -- trim it and try again."
    return None


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    if not _JOB_ID_RE.match(job_id):
        return HTMLResponse("not found", status_code=404)

    record = queue_instance.status(job_id)
    if record is None:
        return templates.TemplateResponse(
            request,
            "error.html",
            {"message": "Unknown job. It may have expired, or the server restarted."},
            status_code=404,
        )

    if record.status == "done":
        return RedirectResponse(f"/reports/{job_id}")

    if record.status == "failed":
        return templates.TemplateResponse(
            request, "error.html", {"message": record.error}, status_code=200
        )

    return templates.TemplateResponse(request, "status.html", {"status": record.status})


@app.get("/reports/{job_id}")
def report(job_id: str):
    if not _JOB_ID_RE.match(job_id):
        return HTMLResponse("not found", status_code=404)
    path = report_path(config.STORAGE_DIR, job_id)
    if not path.exists():
        return HTMLResponse(
            "Report not found -- it may have expired "
            f"({int(config.EXPIRY_HOURS)}h after processing) or never existed.",
            status_code=404,
        )
    return FileResponse(path, media_type="text/html")
