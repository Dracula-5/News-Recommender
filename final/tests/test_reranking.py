import numpy as np

from reranking import mmr_select


def _item(news_id, score, category):
    return {"news_id": news_id, "score": score, "category": category}


def test_returns_empty_for_empty_input():
    assert mmr_select([], k=5) == []


def test_respects_k():
    items = [_item(f"n{i}", 1.0 - i * 0.01, "tech") for i in range(10)]
    selected = mmr_select(items, k=4, lambda_param=1.0)
    assert len(selected) == 4


def test_lambda_one_reduces_to_pure_relevance_ranking():
    items = [_item("a", 0.9, "tech"), _item("b", 0.5, "tech"), _item("c", 0.1, "tech")]
    # Identical (orthogonal-distance-irrelevant) vectors so similarity can't
    # influence the outcome — with lambda=1 only relevance should matter.
    vectors = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "c": np.array([1.0, 0.0])}
    selected = mmr_select(items, k=3, vectors=vectors, lambda_param=1.0)
    assert [it["news_id"] for it in selected] == ["a", "b", "c"]


def test_diversity_prefers_dissimilar_item_over_near_duplicate():
    # "b" is a near-duplicate of top-ranked "a" (same vector); "c" is
    # lower-scored but orthogonal (unrelated content). A diversity-leaning
    # lambda should pick "c" over "b" for the second slot.
    items = [_item("a", 0.95, "tech"), _item("b", 0.90, "tech"), _item("c", 0.60, "world")]
    vectors = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([1.0, 0.0]),
        "c": np.array([0.0, 1.0]),
    }
    selected = mmr_select(items, k=2, vectors=vectors, lambda_param=0.3)
    ids = [it["news_id"] for it in selected]
    assert ids[0] == "a"
    assert ids[1] == "c"


def test_falls_back_to_same_category_similarity_when_no_vectors():
    items = [_item("a", 0.9, "tech"), _item("b", 0.85, "tech"), _item("c", 0.5, "world")]
    selected = mmr_select(items, k=2, vectors={}, lambda_param=0.2)
    ids = [it["news_id"] for it in selected]
    assert ids[0] == "a"
    # "c" (different category => similarity 0) should beat "b" (same
    # category as "a" => similarity 1) once diversity is weighted heavily.
    assert ids[1] == "c"
