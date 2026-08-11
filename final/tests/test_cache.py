import time

from cache import TTLCache


def test_get_returns_none_for_missing_key():
    cache = TTLCache(default_ttl=10)
    assert cache.get("missing") is None


def test_set_then_get_roundtrips():
    cache = TTLCache(default_ttl=10)
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}


def test_entry_expires_after_ttl():
    cache = TTLCache(default_ttl=0.05)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    time.sleep(0.08)
    assert cache.get("k") is None


def test_get_or_set_only_calls_factory_once_within_ttl():
    cache = TTLCache(default_ttl=10)
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return calls["n"]

    first = cache.get_or_set("k", factory)
    second = cache.get_or_set("k", factory)
    assert first == second == 1
    assert calls["n"] == 1


def test_invalidate_forces_recompute():
    cache = TTLCache(default_ttl=10)
    cache.set("k", "old")
    cache.invalidate("k")
    assert cache.get("k") is None


def test_invalidate_prefix_clears_matching_keys_only():
    cache = TTLCache(default_ttl=10)
    cache.set("users:1", "a")
    cache.set("users:2", "b")
    cache.set("categories", "c")
    cache.invalidate_prefix("users:")
    assert cache.get("users:1") is None
    assert cache.get("users:2") is None
    assert cache.get("categories") == "c"
