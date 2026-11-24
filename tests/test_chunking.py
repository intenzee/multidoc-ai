from backend.ingestion.chunking import chunk_text, split_table


def test_empty_text_gives_no_chunks():
    assert chunk_text("", 800, 150) == []
    assert chunk_text("   \n\n ", 800, 150) == []


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", 800, 150) == ["Hello world."]


def test_long_text_respects_size_and_overlaps():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(200)]
    chunks = chunk_text(" ".join(sentences), 300, 60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # consecutive chunks share some text (overlap keeps context across boundaries)
    assert any(chunks[i][-30:].split()[-1] in chunks[i + 1] for i in range(len(chunks) - 1))
    # nothing is lost: every sentence appears in some chunk
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_split_table_repeats_header():
    header = "| Region | Q1 |\n| --- | --- |"
    rows = "\n".join(f"| Region{i} | {i} |" for i in range(100))
    parts = split_table(header + "\n" + rows, 300)
    assert len(parts) > 1
    assert all(p.startswith(header) for p in parts)
