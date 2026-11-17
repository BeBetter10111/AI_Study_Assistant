"""Hierarchical legal chunking (Phần > Chương > Mục > Điều > Khoản)."""
from app.rag.legal_parser import chunk_legal_text, is_legal_document, parse_legal

LAW = """BỘ LUẬT MẪU

Chương I
NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Luật này quy định về quyền và nghĩa vụ của công dân.

Điều 2. Đối tượng áp dụng
1. Công dân Việt Nam.
2. Người nước ngoài cư trú hợp pháp, trừ trường hợp quy định tại khoản 3 Điều này.
3. Người đang bị hạn chế năng lực hành vi không thuộc đối tượng áp dụng.

Chương II
QUYỀN SỞ HỮU

Mục 1. Quy định chung

Điều 3. Quyền sở hữu
Chủ sở hữu có quyền chiếm hữu, sử dụng và định đoạt tài sản.
"""


def test_detects_legal_structure_and_rejects_prose():
    assert is_legal_document(LAW)
    assert not is_legal_document("Đây chỉ là một đoạn văn bình thường.\nKhông có điều khoản nào cả.")


def test_parse_builds_hierarchy_and_clauses():
    doc = parse_legal(LAW)
    assert [a.number for a in doc.articles] == ["1", "2", "3"]
    art2 = doc.articles[1]
    assert [c.number for c in art2.clauses] == ["1", "2", "3"]
    assert "Chương I" in art2.hierarchy_path and "Điều 2" in art2.hierarchy_path
    assert any("Mục 1" in h for h in doc.articles[2].hierarchy)


def test_short_articles_are_one_chunk_each_with_metadata():
    chunks = chunk_legal_text(LAW)
    articles = [c for c in chunks if c.extra.get("article")]
    assert len(articles) == 3
    art2 = next(c for c in articles if "Điều 2" in c.content)
    assert "trừ trường hợp" in art2.content and "khoản 3" in art2.content  # exception stays with the rule
    assert art2.extra["domain"] == "legal"
    assert art2.extra["article_key"]
    assert "Chương I" in art2.extra["hierarchy_path"]


def test_long_article_splits_by_clause_and_repeats_header():
    clauses = "\n".join(f"{i}. " + ("Nội dung khoản rất dài. " * 25) for i in range(1, 7))
    text = f"Chương I\nCHUNG\n\nĐiều 1. Điều dài\nCác quy định sau đây áp dụng:\n{clauses}\n\nĐiều 2. Ngắn\nNội dung ngắn.\n\nĐiều 3. Khác\nNội dung khác.\n"
    chunks = chunk_legal_text(text, max_tokens=120, overlap=10)
    art1 = [c for c in chunks if c.extra.get("article") and c.extra.get("clause")]
    assert len({c.extra["clause"] for c in art1}) >= 3
    assert all("Điều 1" in c.content for c in art1)
    assert len({c.extra["article_key"] for c in art1}) == 1  # siblings share a key for expansion
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_no_articles_returns_empty():
    assert chunk_legal_text("Chỉ là văn bản thường, không có điều luật.") == []


def test_english_article_headings():
    text = "Chapter 1\nGeneral\n\nArticle 1. Scope\nThis Act applies to all.\n\nArticle 2. Terms\nTerms apply.\n\nArticle 3. Duties\nDuties apply.\n"
    doc = parse_legal(text)
    assert [a.number for a in doc.articles] == ["1", "2", "3"]
