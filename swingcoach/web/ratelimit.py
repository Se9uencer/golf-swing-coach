"""Per-IP fixed-window rate limiting.

In-process and in-memory -- correct for the single-instance deployment
this ships with (render.yaml runs one web service, no horizontal scaling),
not for a multi-instance fleet where each instance would count separately.
Good enough for the abuse control this is actually for: a script hammering
uploads from one address, not a distributed one.
"""

import threading
import time


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 3600.0) -> None:
        self._limit = limit
        self._window_s = window_s
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record a hit for `key` and return whether it's within the limit."""
        now = time.time() if now is None else now
        cutoff = now - self._window_s
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if t > cutoff]
            if len(hits) >= self._limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True
