"""STEM formula extraction: math-page detection, backend selection, fail-open behaviour."""
from app.rag import formula_extraction as fx


PROSE = [("TimesNewRoman", "Hệ điều hành là phần mềm quản lý tài nguyên phần cứng và cung cấp dịch vụ.")]
MATH = [("CMMI10", "x"), ("CMSY10", "∑"), ("CMR10", "="), ("MathematicalPi", "∫")]


def test_prose_scores_low_and_math_scores_high():
    assert fx.math_score_from_spans(PROSE) < 0.15
    assert fx.math_score_from_spans(MATH) >= 0.15
    assert fx.math_score_from_spans([]) == 0.0


def test_unicode_math_symbols_raise_score():
    assert fx.math_score_from_spans([("Arial", "∑∫√≤≥ ±∞ α β")]) > 0.15


def test_split_paginated_markdown_and_fence_cleanup():
    pages = fx.split_paginated_markdown("\n\n{0}------------------------------------------------\n\nTrang 1 $x^2$\n\n{1}------------------------------------------------\n\nTrang 2\n")
    assert set(pages) == {0, 1} and "x^2" in pages[0]
    assert fx._clean_markdown("```markdown\n$$a^2+b^2=c^2$$\n```") == "$$a^2+b^2=c^2$$"


def test_get_extractor_off_by_default_and_unknown_backend(monkeypatch):
    monkeypatch.setattr(fx.settings, "stem_extraction_backend", "none")
    assert fx.get_extractor() is None
    monkeypatch.setattr(fx.settings, "stem_extraction_backend", "does-not-exist")
    assert fx.get_extractor() is None
    monkeypatch.setattr(fx.settings, "stem_extraction_backend", "vlm")
    assert isinstance(fx.get_extractor(), fx.VLMExtractor)


class _Fake:
    name = "fake"

    def __init__(self, out=None, boom=False):
        self.out, self.boom, self.asked = out or {}, boom, None

    def extract(self, file_path, page_indices):
        self.asked = list(page_indices)
        if self.boom:
            raise RuntimeError("backend down")
        return self.out


def test_extract_formula_pages_uses_detected_pages_and_filters_output(monkeypatch):
    monkeypatch.setattr(fx, "detect_math_pages", lambda path, threshold=None: [1, 3])
    monkeypatch.setattr(fx.settings, "stem_extract_all_pages", False)
    fake = _Fake({1: r"$\frac{a}{b}$", 3: "   ", 9: "out of range"})
    result = fx.extract_formula_pages("x.pdf", 5, extractor=fake)
    assert fake.asked == [1, 3]
    assert result == {1: r"$\frac{a}{b}$"}


def test_extract_formula_pages_fails_open(monkeypatch):
    monkeypatch.setattr(fx, "detect_math_pages", lambda path, threshold=None: [0])
    assert fx.extract_formula_pages("x.pdf", 2, extractor=_Fake(boom=True)) == {}
    monkeypatch.setattr(fx.settings, "stem_extraction_backend", "none")
    assert fx.extract_formula_pages("x.pdf", 2) == {}


def test_vlm_page_cap(monkeypatch):
    monkeypatch.setattr(fx, "detect_math_pages", lambda path, threshold=None: list(range(10)))
    monkeypatch.setattr(fx.settings, "vlm_max_pages", 3)
    fake = _Fake({0: "a"})
    fx.extract_formula_pages("x.pdf", 10, extractor=fake)
    assert fake.asked == [0, 1, 2]
