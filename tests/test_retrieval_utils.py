from backend.retrieval.retriever import (EvidenceItem, is_overview, reciprocal_rank_fusion,
                                        spread_over_pages, tokenize)


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    ranking = sorted(fused, key=fused.get, reverse=True)
    assert set(ranking[:2]) == {"a", "b"}
    assert fused["a"] == fused["b"]
    assert fused["c"] < fused["a"] and fused["d"] < fused["a"]


def test_tokenize_keeps_numbers_and_drops_stopwords():
    assert tokenize("What was the revenue in Q3 2025? It was 7.8 crore.") == [
        "revenue", "q3", "2025", "7.8", "crore"]


def test_overview_questions_are_detected():
    for q in ["Summarise the main points of the documents.", "TELL ME SUBJECT NAMES",
              "the name of courses", "What is this document about?", "Give an overview"]:
        assert is_overview(q), q
    for q in ["What is the warranty period?", "Who is the CEO of Tesla?",
              "How many employees did GreenLeaf have at the end of 2025?"]:
        assert not is_overview(q), q


def test_spread_over_pages_puts_one_item_per_page_first():
    def item(i, page):
        return EvidenceItem(id=i, kind="text", content_type="text", document_id="d",
                            document_name="d.pdf", page_number=page, text=i)
    ranked = [item("a", 1), item("b", 1), item("c", 2), item("d", 1), item("e", 3)]
    assert [i.id for i in spread_over_pages(ranked)] == ["a", "c", "e", "b", "d"]
