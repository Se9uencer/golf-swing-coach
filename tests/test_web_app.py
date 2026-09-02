"""Smoke tests for the FastAPI routes that don't require actually running
the pose pipeline (no model file is committed to the repo, see
models/README.md). These cover request validation and job-id handling;
`swingcoach.web.pipeline` and the analysis modules it wraps have their own
tests and are exercised for real via the CLI against real footage per
AGENTS.md -- this file is not a substitute for that.
"""

from fastapi.testclient import TestClient

from swingcoach.web import config
from swingcoach.web.app import app


def client_with_ip(ip: str) -> TestClient:
    """A TestClient pinned to a distinct fake IP via X-Forwarded-For, so
    each test gets its own rate-limit bucket in the app's shared, in-process
    limiter and tests don't interfere with each other's counts.
    """
    c = TestClient(app)
    c.headers.update({"X-Forwarded-For": ip})
    return c


def test_index_ok():
    with client_with_ip("10.0.0.1") as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "swingcoach" in r.text.lower()


def test_healthz_ok():
    with client_with_ip("10.0.0.2") as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_upload_rejects_unsupported_extension():
    with client_with_ip("10.0.0.3") as c:
        r = c.post("/upload", files={"video": ("clip.txt", b"not a video", "text/plain")})
        assert r.status_code == 400


def test_upload_rejects_unreadable_video_content():
    with client_with_ip("10.0.0.4") as c:
        r = c.post("/upload", files={"video": ("clip.mp4", b"garbage bytes", "video/mp4")})
        assert r.status_code == 400


def test_upload_enforces_rate_limit():
    with client_with_ip("10.0.0.5") as c:
        for _ in range(config.RATE_LIMIT_PER_HOUR):
            r = c.post("/upload", files={"video": ("clip.txt", b"x", "text/plain")})
            assert r.status_code == 400  # bad extension, but consumes the limit
        r = c.post("/upload", files={"video": ("clip.txt", b"x", "text/plain")})
        assert r.status_code == 429


def test_unknown_job_id_is_404():
    with client_with_ip("10.0.0.6") as c:
        r = c.get("/jobs/deadbeefdeadbeefdeadbeefdeadbeef")
        assert r.status_code == 404


def test_malformed_job_id_is_404():
    with client_with_ip("10.0.0.7") as c:
        r = c.get("/jobs/../../etc/passwd")
        assert r.status_code == 404
        r = c.get("/reports/not-a-hex-id")
        assert r.status_code == 404
