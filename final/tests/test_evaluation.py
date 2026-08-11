from evaluation import (
    catalog_coverage,
    category_diversity,
    ndcg_at_k,
    precision_at_k,
)


def _items(categories):
    return [{"news_id": f"n{i}", "category": c} for i, c in enumerate(categories)]


def test_precision_at_k_all_relevant():
    items = _items(["tech", "tech", "sports"])
    assert precision_at_k(items, {"tech", "sports"}) == 1.0


def test_precision_at_k_none_relevant():
    items = _items(["politics", "world"])
    assert precision_at_k(items, {"tech"}) == 0.0


def test_precision_at_k_partial():
    items = _items(["tech", "world", "tech", "world"])
    assert precision_at_k(items, {"tech"}) == 0.5


def test_precision_at_k_empty_items():
    assert precision_at_k([], {"tech"}) == 0.0


def test_ndcg_perfect_ranking_scores_one():
    # All relevant items already at the top -> DCG == IDCG.
    items = _items(["tech", "tech", "world"])
    assert ndcg_at_k(items, {"tech"}) == 1.0


def test_ndcg_rewards_relevant_items_ranked_higher():
    relevant_first = _items(["tech", "world"])
    relevant_last = _items(["world", "tech"])
    assert ndcg_at_k(relevant_first, {"tech"}) > ndcg_at_k(relevant_last, {"tech"})


def test_category_diversity_all_same_category():
    items = _items(["tech", "tech", "tech"])
    assert category_diversity(items) == 1 / 3


def test_category_diversity_all_distinct():
    items = _items(["tech", "sports", "world"])
    assert category_diversity(items) == 1.0


def test_category_diversity_empty():
    assert category_diversity([]) == 0.0


def test_catalog_coverage_ratio():
    assert catalog_coverage({"a", "b"}, catalog_size=4) == 0.5


def test_catalog_coverage_empty_catalog_is_zero_not_divide_by_zero():
    assert catalog_coverage(set(), catalog_size=0) == 0.0
