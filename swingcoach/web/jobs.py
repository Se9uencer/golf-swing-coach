"""In-process job queue.

No Redis, no separate worker service: a bounded queue.Queue feeding
`WORKER_CONCURRENCY` daemon threads inside the same process as the FastAPI
app. That's a deliberate simplification, not an oversight -- Render
persistent disks attach to a single service, so a separate web + worker
service pair would need shared object storage (S3-compatible) just to
hand the uploaded file from one to the other. One service, one disk, one
process avoids that entirely. See PLAN.md section 10 for the tradeoff
(job state lives in memory and is lost on restart; the uploaded file and
any finished report survive on disk regardless).
"""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from swingcoach.ingest import VideoLoadError
from swingcoach.pose import PoseModelMissing
from swingcoach.web import config
from swingcoach.web.pipeline import NoSwingsDetected, process
from swingcoach.web.storage import job_dir

logger = logging.getLogger(__name__)


class QueueFull(RuntimeError):
    pass


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"  # queued -> processing -> done | failed
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class JobQueue:
    def __init__(self, storage_dir: Path, concurrency: int, max_depth: int) -> None:
        self._storage_dir = storage_dir
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_depth)
        self._records: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._threads = [
            threading.Thread(target=self._worker_loop, daemon=True, name=f"swingcoach-worker-{i}")
            for i in range(concurrency)
        ]
        self._started = False

    def start(self) -> None:
        """Idempotent: the FastAPI lifespan can run more than once against
        this same process-wide singleton (e.g. one TestClient per test).
        """
        with self._lock:
            if self._started:
                return
            self._started = True
        for t in self._threads:
            t.start()

    def submit(self, job_id: str, source_name: str) -> None:
        with self._lock:
            self._records[job_id] = JobRecord(job_id=job_id)
        try:
            self._queue.put_nowait((job_id, source_name))
        except queue.Full as e:
            with self._lock:
                del self._records[job_id]
            raise QueueFull("the queue is full right now -- try again in a few minutes") from e

    def status(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def _worker_loop(self) -> None:
        while True:
            job_id, source_name = self._queue.get()
            with self._lock:
                record = self._records.get(job_id)
                if record:
                    record.status = "processing"
            try:
                process(self._storage_dir, job_id, source_name)
                with self._lock:
                    self._records[job_id].status = "done"
            except (VideoLoadError, PoseModelMissing, NoSwingsDetected) as e:
                self._fail(job_id, str(e))
            except Exception:
                logger.exception("job %s failed", job_id)
                self._fail(job_id, "something went wrong processing this video")
            finally:
                self._queue.task_done()

    def _fail(self, job_id: str, message: str) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record:
                record.status = "failed"
                record.error = message


def new_job_id() -> str:
    return uuid.uuid4().hex


def make_job_dir(storage_dir: Path, job_id: str) -> Path:
    d = job_dir(storage_dir, job_id)
    d.mkdir(parents=True, exist_ok=False)
    return d


queue_instance = JobQueue(
    storage_dir=config.STORAGE_DIR,
    concurrency=config.WORKER_CONCURRENCY,
    max_depth=config.MAX_QUEUE_DEPTH,
)
