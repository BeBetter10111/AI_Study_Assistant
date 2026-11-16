"""Unit tests for GraphRAG extraction / formatting (no live DB or LLM)."""
from app.rag.chunking import ChunkData
from app.rag import knowledge_graph as kg


def test_extract_heuristic_english_is_a():
    chunks = [ChunkData(content="Photosynthesis is a process that converts light energy.")]
    graph = kg.extract_heuristic(chunks)
    assert graph.triples
    assert any(t.relation == "is_a" for t in graph.triples)
    assert any("Photosynthesis" in t.source for t in graph.triples)


def test_extract_heuristic_vietnamese_causes():
    chunks = [ChunkData(content="Hút thuốc gây ra ung thư phổi ở nhiều bệnh nhân.")]
    graph = kg.extract_heuristic(chunks)
    assert any(t.relation == "causes" for t in graph.triples)


def test_extract_skips_summary_chunks():
    chunks = [
        ChunkData(content="Summary only", extra={"is_summary": True}),
        ChunkData(content="Insulin is a hormone that regulates blood sugar."),
    ]
    graph = kg.extract_heuristic(chunks)
    assert all(t.chunk_index == 1 for t in graph.triples)


def test_format_facts_block():
    assert kg.format_facts_block([]) == ""
    block = kg.format_facts_block(["A --is_a--> B"])
    assert "Knowledge-graph relations" in block
    assert "A --is_a--> B" in block


def test_triple_normalise_rejects_junk():
    assert kg.Triple("a", "is_a", "a").normalised() is None
    assert kg.Triple("the", "is_a", "and").normalised() is None
    ok = kg.Triple("Neural Network", "is_a", "Machine Learning Model").normalised()
    assert ok is not None
    assert ok.relation == "is_a"
