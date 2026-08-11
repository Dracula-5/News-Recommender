import time

from ratelimit import RateLimiter


def test_allows_requests_under_the_limit():
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert limiter.allow("client-a") is True


def test_blocks_requests_over_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False


def test_clients_are_isolated():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True
    assert limiter.allow("client-a") is False


def test_window_resets_after_expiry():
    limiter = RateLimiter(max_requests=1, window_seconds=0.05)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False
    time.sleep(0.08)
    assert limiter.allow("client-a") is True
