from backend.retrieval.retriever import reciprocal_rank_fusion, tokenize


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    ranking = sorted(fused, key=fused.get, reverse=True)
    assert set(ranking[:2]) == {"a", "b"}
    assert fused["a"] == fused["b"]
    assert fused["c"] < fused["a"] and fused["d"] < fused["a"]


def test_tokenize_keeps_numbers_and_drops_stopwords():
    assert tokenize("What was the revenue in Q3 2025? It was 7.8 crore.") == [
        "revenue", "q3", "2025", "7.8", "crore"]
