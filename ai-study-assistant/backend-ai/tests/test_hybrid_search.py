"""Logic thuần của hybrid search (app/rag/hybrid.py) — không cần Qdrant/model."""
from app.rag import hybrid


def _overlap(query: str, doc: str) -> int:
    q, _ = hybrid.sparse_query_vector(query)
    d, _ = hybrid.sparse_document_vector(doc)
    return len(set(q) & set(d))


def test_fold_diacritics_handles_vietnamese_d():
    assert hybrid.fold_diacritics("Hệ điều hành Đại học") == "He dieu hanh Dai hoc"


def test_accentless_query_matches_accented_text():
    doc = "Hệ điều hành quản lý tài nguyên phần cứng"
    assert _overlap("he dieu hanh", doc) > 0
    # exact accents match in more namespaces than accent-less typing
    assert _overlap("hệ điều hành", doc) > _overlap("he dieu hanh", doc)


def test_exact_code_is_a_term():
    assert _overlap("IT069IU", "Mã môn học IT069IU: OOP") >= 2


def test_stopword_only_query_gives_empty_vector():
    assert hybrid.sparse_query_vector("the of is và là") == ([], [])


def test_term_index_is_stable_and_positive():
    assert hybrid.term_index("database") == hybrid.term_index("database")
    assert 0 <= hybrid.term_index("database") < 2**31


def test_document_vector_sorted_unique_and_longer_text_weighs_less():
    idx, val = hybrid.sparse_document_vector("sql sql sql join")
    assert idx == sorted(set(idx)) and len(idx) == len(val)
    i = hybrid.term_index("sql")
    short = dict(zip(*hybrid.sparse_document_vector("sql")))
    long_ = dict(zip(*hybrid.sparse_document_vector("sql " + "filler " * 300)))
    assert short[i] > long_[i]


def test_weighted_rrf_rewards_agreement_and_respects_weights():
    s = hybrid.weighted_rrf([(0.5, ["a", "b"]), (0.5, ["b", "c"])], k=60)
    assert s["b"] > s["a"] and s["b"] > s["c"]
    s = hybrid.weighted_rrf([(1.0, ["a"]), (0.0, ["b"])])
    assert "b" not in s


def test_minmax():
    assert hybrid.minmax([1, 2, 3]) == [0.0, 0.5, 1.0]
    assert hybrid.minmax([5, 5]) == [1.0, 1.0]
    assert hybrid.minmax([]) == []


def _hit(content, summary=False, **kw):
    return {"document_id": "d", "content": content, "is_summary": summary, **kw}


def test_summary_dropped_only_when_its_detail_chunk_is_present():
    detail = _hit("Intro sentence. More detail here.", page_number=1)
    summ = _hit("Intro sentence.", summary=True, page_number=1)
    other_page = _hit("Intro sentence.", summary=True, page_number=2)
    assert hybrid.drop_redundant_summaries([summ, detail]) == [detail]
    assert hybrid.drop_redundant_summaries([summ]) == [summ]
    assert other_page in hybrid.drop_redundant_summaries([other_page, detail])
