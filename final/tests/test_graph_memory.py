"""
Regression tests for the graph_memory.py bugs found and fixed in this pass:
  1. get_co_interest_categories() deadlocked (called a lock-acquiring
     public method from inside another lock-acquiring public method, on a
     non-reentrant threading.Lock).
  2. _load_from_db() double-decayed: it applied decay once at load time but
     left `last_time` at the *original* stale timestamp, so the next
     record_interaction() decayed the same elapsed window a second time.
  3. Interest score averaged decayed total_reward over a raw, never-decayed
     `count`, so long-history users' scores sank toward zero over time
     even with steady engagement.
  4. Co-interest edges were never persisted (lost on every restart) and
     never decayed (grew forever once created).
"""
from __future__ import annotations

import time

import pytest

from graph_memory import UserInterestGraph, _HALF_LIFE


@pytest.fixture
def graph(db_path):
    return UserInterestGraph(db_path)


def test_co_interest_categories_does_not_deadlock_on_empty_graph(graph):
    # Regression test for bug #1 — this used to hang forever.
    result = graph.get_co_interest_categories("u1", "technology", n=3)
    assert result == []


def test_co_interest_categories_does_not_deadlock_with_data(graph):
    graph.record_interaction("u1", "technology", 3.0, True, 10)
    graph.record_interaction("u1", "science", 3.0, True, 10)
    result = graph.get_co_interest_categories("u1", "technology", n=3)
    assert result == ["science"]


def test_liking_two_categories_creates_a_co_interest_edge(graph):
    graph.record_interaction("u1", "technology", 3.0, True, 10)
    graph.record_interaction("u1", "science", 3.0, True, 10)
    stats = graph.get_graph_stats("u1")
    pairs = {(e["a"], e["b"]) for e in stats["co_interest_edges"]}
    assert ("technology", "science") in pairs
    assert ("science", "technology") in pairs


def test_score_decays_over_time(graph, monkeypatch):
    t0 = 1_000_000.0
    monkeypatch.setattr(time, "time", lambda: t0)
    graph.record_interaction("u1", "technology", 4.0, True, 10)
    fresh_score = graph.get_category_score("u1", "technology")

    # One half-life later, with no new interaction.
    monkeypatch.setattr(time, "time", lambda: t0 + _HALF_LIFE)
    decayed_score = graph.get_category_score("u1", "technology")

    assert decayed_score < fresh_score


def test_reload_does_not_double_decay(graph, db_path, monkeypatch):
    """
    Regression test for bug #2: save, advance the clock, reload into a
    fresh instance, then advance the clock by the *same* amount again and
    compare against a reference graph that only ever decayed once for that
    combined elapsed time. Before the fix, the reload+second-wait path
    decayed measurably more than the single continuous wait.
    """
    t0 = 1_000_000.0
    monkeypatch.setattr(time, "time", lambda: t0)
    graph.record_interaction("u1", "technology", 4.0, True, 10)
    graph.save_to_db("u1")

    elapsed = _HALF_LIFE * 0.5

    # Reference: one continuous graph, decayed once for 2x elapsed.
    monkeypatch.setattr(time, "time", lambda: t0 + 2 * elapsed)
    reference_score = graph.get_category_score("u1", "technology")

    # Reload path: save at t0, "restart" (fresh instance + load) at
    # t0+elapsed, then read again at t0+2*elapsed.
    monkeypatch.setattr(time, "time", lambda: t0 + elapsed)
    reloaded = UserInterestGraph(db_path)
    monkeypatch.setattr(time, "time", lambda: t0 + 2 * elapsed)
    reload_score = reloaded.get_category_score("u1", "technology")

    assert reload_score == pytest.approx(reference_score, rel=0.02)


def test_long_history_does_not_sink_score_to_zero_with_steady_engagement(graph, monkeypatch):
    """
    Regression test for bug #3: with total_reward/decayed_count (rather
    than total_reward/raw_count), a user who keeps liking a category at a
    steady rate should keep a roughly stable score, not one that trends
    toward zero purely because `count` grew large over a long history.
    """
    t = 1_000_000.0
    monkeypatch.setattr(time, "time", lambda: t)
    for _ in range(50):
        graph.record_interaction("u1", "technology", 3.0, True, 10)
        t += 3600  # one interaction per hour, well within the half-life
        monkeypatch.setattr(time, "time", lambda: t)

    score = graph.get_category_score("u1", "technology")
    assert score > 0.7  # consistently positive engagement should stay confidently positive


def test_co_interest_edges_persist_across_reload(graph, db_path):
    graph.record_interaction("u1", "technology", 3.0, True, 10)
    graph.record_interaction("u1", "science", 3.0, True, 10)
    graph.save_to_db("u1")

    reloaded = UserInterestGraph(db_path)
    assert reloaded.get_co_interest_categories("u1", "technology") == ["science"]


def test_get_top_categories_ranks_by_score(graph):
    graph.record_interaction("u1", "technology", 4.0, True, 15)
    graph.record_interaction("u1", "sports", -3.0, False, 1)
    top = graph.get_top_categories("u1", n=2)
    assert top[0] == "technology"
