"""Unit tests for the per-IP fixed-window rate limiter."""

from swingcoach.web.ratelimit import RateLimiter


def test_allows_up_to_limit():
    limiter = RateLimiter(limit=3, window_s=3600.0)
    assert limiter.allow("1.2.3.4", now=0.0)
    assert limiter.allow("1.2.3.4", now=1.0)
    assert limiter.allow("1.2.3.4", now=2.0)
    assert not limiter.allow("1.2.3.4", now=3.0)


def test_keys_are_independent():
    limiter = RateLimiter(limit=1, window_s=3600.0)
    assert limiter.allow("a", now=0.0)
    assert limiter.allow("b", now=0.0)
    assert not limiter.allow("a", now=1.0)


def test_window_expires():
    limiter = RateLimiter(limit=1, window_s=60.0)
    assert limiter.allow("a", now=0.0)
    assert not limiter.allow("a", now=30.0)
    assert limiter.allow("a", now=61.0)
