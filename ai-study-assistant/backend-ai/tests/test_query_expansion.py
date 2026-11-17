"""Entity linking / query expansion across the bundled domain lexicons."""
import pytest

from app.rag import query_expansion as qe


@pytest.fixture(autouse=True)
def _fresh_linkers(monkeypatch):
    monkeypatch.setattr(qe.settings, "query_expansion_enabled", True)
    monkeypatch.setattr(qe.settings, "query_expansion_max_terms", 8)
    monkeypatch.setattr(qe.settings, "umls_api_key", "", raising=False)
    monkeypatch.setattr(qe.settings, "medical_lexicon_paths", "")
    qe.reset_linkers()
    yield
    qe.reset_linkers()


def _terms(question):
    return [t.lower() for t in qe.expand_query(question).terms]


def test_brand_name_expands_to_active_ingredient():
    assert "paracetamol" in _terms("Liều dùng Panadol cho người lớn?")


def test_matching_ignores_vietnamese_diacritics_and_case():
    assert "paracetamol" in _terms("tac dung phu cua PANADOL")


def test_science_lexicon_is_loaded():
    assert "dna" in _terms("Cấu trúc của ADN là gì?")
    assert "photosynthesis" in _terms("Giải thích quá trình quang hợp")


def test_law_and_economics_lexicon_is_loaded():
    assert "inflation" in _terms("Lạm phát ảnh hưởng thế nào đến lãi suất?")
    assert "statute of limitations" in _terms("Thời hiệu khởi kiện là bao lâu?")


def test_no_terms_for_plain_question_and_no_echo_of_query_words():
    assert qe.expand_query("Hôm nay trời đẹp quá").terms == []
    assert "paracetamol" not in _terms("What is paracetamol?")  # already in the query


def test_max_terms_cap_and_disabled_flag(monkeypatch):
    assert len(qe.expand_query("Panadol Advil Aspirin", max_terms=2).terms) <= 2
    monkeypatch.setattr(qe.settings, "query_expansion_enabled", False)
    assert qe.expand_query("Panadol").terms == []


def test_expand_query_fails_open(monkeypatch):
    def boom(_q):
        raise RuntimeError("lexicon exploded")
    monkeypatch.setattr(qe, "link_entities", boom)
    assert qe.expand_query("Panadol").terms == []


def test_extra_csv_lexicon(tmp_path, monkeypatch):
    path = tmp_path / "extra.csv"
    path.write_text("canonical,synonyms,type,id\nmetformin,Glucophage|Diaformin,drug,MeSH:D008687\n", encoding="utf-8")
    monkeypatch.setattr(qe.settings, "medical_lexicon_paths", str(path))
    qe.reset_linkers()
    assert "metformin" in _terms("Tác dụng của Glucophage")
