"""
evaluation/metrics.py

Retrieval metrics at PAGE level: a retrieved item counts as relevant if its
(document, page) is in the question's relevant set. Several chunks from the same
relevant page are counted once (duplicates are removed before scoring), so a
system cannot score well by repeating one page.
"""

import re


def dedupe(ranked: list[tuple[str, int]]) -> list[tuple[str, int]]:
    seen, out = set(), []
    for key in ranked:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def precision_at_k(ranked, relevant, k: int) -> float:
    top = dedupe(ranked)[:k]
    return sum(1 for r in top if r in relevant) / k if k else 0.0


def recall_at_k(ranked, relevant, k: int) -> float:
    if not relevant:
        return 0.0
    top = dedupe(ranked)[:k]
    return len(set(top) & set(relevant)) / len(set(relevant))


def hit_at_k(ranked, relevant, k: int) -> float:
    return 1.0 if set(dedupe(ranked)[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranked, relevant) -> float:
    for rank, key in enumerate(dedupe(ranked), start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred or not ref:
        return 0.0
    common = sum(min(pred.count(t), ref.count(t)) for t in set(pred))
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    text = answer.lower()
    return sum(1 for k in keywords if k.lower() in text) / len(keywords)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
