from evaluation.metrics import (hit_at_k, keyword_score, precision_at_k, recall_at_k,
                                reciprocal_rank, token_f1)

A, B, C = ("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1)


def test_rank_metrics():
    ranked = [B, B, A, C]  # duplicate page B is counted once
    relevant = {A}
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert hit_at_k(ranked, relevant, 1) == 0.0
    assert hit_at_k(ranked, relevant, 2) == 1.0
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, {A, C}, 2) == 0.5
    assert recall_at_k(ranked, {A, C}, 3) == 1.0


def test_no_relevant_pages():
    assert reciprocal_rank([A], set()) == 0.0
    assert recall_at_k([A], set(), 3) == 0.0


def test_answer_scores():
    assert keyword_score("Revenue was 4.9 crore", ["4.9"]) == 1.0
    assert keyword_score("Pune only", ["Pune", "Rotterdam"]) == 0.5
    assert token_f1("the revenue was 4.9", "revenue 4.9") > 0.6
    assert token_f1("", "x") == 0.0
