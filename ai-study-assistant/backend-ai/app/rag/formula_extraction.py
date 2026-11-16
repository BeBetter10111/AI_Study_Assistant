"""Formula-aware PDF extraction for STEM documents (Markdown + LaTeX).

``page.get_text("text")`` flattens ``\\frac{a}{b}``, matrices, roots, sub/superscripts into
garbage.  This module:

1. **Detects** pages that contain mathematics (cheap, PyMuPDF only): math fonts (Computer
   Modern CMMI/CMSY/CMEX, Cambria Math, Symbol, STIX, ...), the density of math Unicode
   characters and how fragmented the text spans are (formulas are typeset glyph by glyph).
2. **Re-extracts** only those pages with a pluggable backend that outputs Markdown with LaTeX:

   * ``marker``  — `marker-pdf <https://github.com/VikParuchuri/marker>`_ (optional pip package)
   * ``nougat``  — Meta's Nougat CLI (optional, run as a subprocess)
   * ``vlm``     — a vision-language model (Qwen2-VL / Qwen2.5-VL, ...) served by Ollama or any
     OpenAI-compatible server (vLLM, ...)

Every backend is optional: if it is not installed / not reachable / fails on a page, the caller
keeps the plain PyMuPDF (or OCR) text for that page.  Nothing here is imported at module load
except PyMuPDF and httpx.
"""
from __future__ import annotations

import base64
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Math detection
_MATH_FONT_HINTS = (
    "cmmi", "cmsy", "cmex", "cmbsy", "msbm", "msam", "eufm", "rsfs", "stmary", "wasy",
    "cambriamath", "cambria math", "symbol", "stix", "mathematical", "euler", "mtpro",
    "mtmi", "mtsy", "lmmath", "latinmodernmath", "asanamath", "xitsmath", "gyremath", "mt extra",
)
_MATH_CHARS = frozenset(
    "∑∏∫∬∭∮√∛∜∂∇∆≤≥≠≈≅≡∝±∓×÷∞∈∉∋⊂⊃⊆⊇⊄∪∩∧∨¬∀∃∄→←↔⇒⇐⇔↦∘⋅∙⊕⊗⊥∥∠°′″‴"
    "αβγδεζηθϑικλμνξπϖρϱσςτυφϕχψωΓΔΘΛΞΠΣΥΦΨΩ"
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ℝℕℤℚℂℏ"
)


def math_score_from_spans(spans: Iterable[Tuple[str, str]]) -> float:
    """Score in [0, 1] from ``(font_name, text)`` pairs — pure function, unit-testable.

    The score is the strongest of three signals so that ANY clear sign of math triggers:
    * share of characters typeset in a math font,
    * density of math Unicode symbols (scaled: 12.5% symbols already counts as 1.0),
    * share of single-glyph spans (formulas are often emitted glyph by glyph), damped by 0.5
      because tables / drop caps also produce short spans.
    """
    total_chars = math_font_chars = math_symbols = 0
    n_spans = single_glyph = 0
    for font, text in spans:
        stripped = "".join((text or "").split())
        if not stripped:
            continue
        n_spans += 1
        total_chars += len(stripped)
        if len(stripped) == 1:
            single_glyph += 1
        f = (font or "").lower().replace("-", "").replace("_", "")
        if any(h.replace(" ", "") in f for h in _MATH_FONT_HINTS):
            math_font_chars += len(stripped)
        math_symbols += sum(1 for ch in stripped if ch in _MATH_CHARS)
    if total_chars == 0:
        return 0.0
    font_ratio = math_font_chars / total_chars
    symbol_ratio = min(1.0, 8.0 * math_symbols / total_chars)
    glyph_ratio = 0.5 * (single_glyph / n_spans) if n_spans >= 20 else 0.0
    return round(max(font_ratio, symbol_ratio, glyph_ratio), 4)


def _page_spans(page) -> List[Tuple[str, str]]:
    spans: List[Tuple[str, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                spans.append((span.get("font", ""), span.get("text", "")))
    return spans


def page_math_score(page) -> float:
    """Math score of a PyMuPDF page (0 = plain prose, 1 = formula-heavy)."""
    try:
        return math_score_from_spans(_page_spans(page))
    except Exception as exc:  # noqa: BLE001 - detection must never break ingestion
        logger.debug(f"Math detection failed on a page: {exc}")
        return 0.0


def detect_math_pages(file_path: str, threshold: Optional[float] = None) -> List[int]:
    """0-based indices of pages that look like they contain mathematics."""
    import fitz

    threshold = settings.stem_math_score_threshold if threshold is None else threshold
    found: List[int] = []
    with fitz.open(file_path) as doc:
        for idx, page in enumerate(doc):
            if page_math_score(page) >= threshold:
                found.append(idx)
    return found


# Backend adapters
class PageMarkdownExtractor(Protocol):
    name: str

    def extract(self, file_path: str, page_indices: Sequence[int]) -> Dict[int, str]:
        """Return ``{page_index: markdown}`` for the pages it could convert (0-based)."""
        ...


_FENCE_RE = re.compile(r"^\s*```(?:markdown|md|latex)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _clean_markdown(text: str) -> str:
    text = (text or "").strip()
    m = _FENCE_RE.match(text)
    return (m.group(1) if m else text).strip()


VLM_PROMPT = (
    "Transcribe this document page to Markdown. Keep the original language and reading order. "
    "Write every mathematical expression in LaTeX: inline math as $...$ and display equations as "
    "$$...$$ (use \\frac, \\sqrt, \\sum, \\int, matrices with \\begin{pmatrix}...\\end{pmatrix}, "
    "chemical formulas with subscripts). Render tables as Markdown tables. "
    "Output ONLY the transcription, no comments."
)


class VLMExtractor:
    """Page image -> Markdown/LaTeX with a vision-language model (Qwen2-VL, Qwen2.5-VL, ...)."""

    name = "vlm"

    def __init__(self, provider: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout_s: Optional[float] = None, dpi: Optional[int] = None):
        self.provider = (provider or settings.vlm_provider).strip().lower()
        default_url = settings.ollama_base_url if self.provider == "ollama" else "https://api.openai.com"
        self.base_url = (base_url or settings.vlm_base_url or default_url).rstrip("/")
        self.model = model or settings.vlm_model
        self.api_key = api_key or settings.vlm_api_key
        self.timeout_s = timeout_s or settings.vlm_timeout_s
        self.dpi = dpi or settings.vlm_dpi

    def _request(self, image_b64: str) -> Tuple[str, dict, dict]:
        """(url, json payload, headers) for one page — separated for testing."""
        if self.provider == "ollama":
            return (
                f"{self.base_url}/api/chat",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": VLM_PROMPT, "images": [image_b64]}],
                    "stream": False,
                    "options": {"temperature": 0},
                },
                {},
            )
        url = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        return (
            f"{url}/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }],
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )

    def _parse(self, data: dict) -> str:
        if self.provider == "ollama":
            return data.get("message", {}).get("content", "")
        return data["choices"][0]["message"]["content"]

    def transcribe_png(self, png_bytes: bytes) -> str:
        url, payload, headers = self._request(base64.b64encode(png_bytes).decode("ascii"))
        response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        response.raise_for_status()
        return _clean_markdown(self._parse(response.json()))

    def extract(self, file_path: str, page_indices: Sequence[int]) -> Dict[int, str]:
        import fitz

        out: Dict[int, str] = {}
        with fitz.open(file_path) as doc:
            for idx in page_indices:
                if idx < 0 or idx >= doc.page_count:
                    continue
                try:
                    png = doc[idx].get_pixmap(dpi=self.dpi).tobytes("png")
                    text = self.transcribe_png(png)
                    if text:
                        out[idx] = text
                except Exception as exc:  # noqa: BLE001 - per-page fallback
                    logger.warning(f"VLM transcription failed on page {idx + 1}: {exc}")
        return out


_PAGE_SPLIT_RE = re.compile(r"\n*\{(\d+)\}-{20,}\n*")
_marker_models = None
_marker_lock = threading.Lock()


def split_paginated_markdown(text: str) -> Dict[int, str]:
    """Split marker's ``paginate_output`` text (``{page_id}------...``) into pages."""
    parts = _PAGE_SPLIT_RE.split(text or "")
    out: Dict[int, str] = {}
    # parts = [before, id0, text0, id1, text1, ...]
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip()
        if body:
            out[int(parts[i])] = body
    return out


class MarkerExtractor:
    """`marker-pdf` (optional dependency) — layout + equation models, outputs Markdown/LaTeX."""

    name = "marker"

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("marker") is not None

    def extract(self, file_path: str, page_indices: Sequence[int]) -> Dict[int, str]:
        global _marker_models
        from marker.converters.pdf import PdfConverter  # type: ignore[import-not-found]
        from marker.models import create_model_dict  # type: ignore[import-not-found]
        from marker.output import text_from_rendered  # type: ignore[import-not-found]

        if _marker_models is None:
            with _marker_lock:
                if _marker_models is None:
                    logger.info("Loading marker-pdf models (first use) …")
                    _marker_models = create_model_dict()
        config = {
            "page_range": ",".join(str(i) for i in sorted(set(page_indices))),
            "paginate_output": True,
            "output_format": "markdown",
            "disable_image_extraction": True,
        }
        converter = PdfConverter(artifact_dict=_marker_models, config=config)
        text, _, _ = text_from_rendered(converter(file_path))
        pages = split_paginated_markdown(text)
        wanted = set(page_indices)
        return {k: _clean_markdown(v) for k, v in pages.items() if k in wanted}


class NougatExtractor:
    """Meta Nougat via its CLI (``pip install nougat-ocr``). One subprocess per page so the output
    maps exactly to a page index (Nougat does not mark page breaks)."""

    name = "nougat"

    def __init__(self, cmd: Optional[str] = None, timeout_s: Optional[float] = None):
        self.cmd = cmd or settings.nougat_cmd
        self.timeout_s = timeout_s or settings.nougat_timeout_s

    def available(self) -> bool:
        return shutil.which(self.cmd) is not None

    def extract(self, file_path: str, page_indices: Sequence[int]) -> Dict[int, str]:
        out: Dict[int, str] = {}
        stem = os.path.splitext(os.path.basename(file_path))[0]
        for idx in page_indices:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    subprocess.run(
                        [self.cmd, file_path, "-o", tmp, "--pages", str(idx + 1), "--markdown", "--no-skipping"],
                        check=True, capture_output=True, timeout=self.timeout_s,
                    )
                    mmd = os.path.join(tmp, f"{stem}.mmd")
                    if os.path.exists(mmd):
                        with open(mmd, "r", encoding="utf-8", errors="replace") as fh:
                            text = _clean_markdown(fh.read())
                        if text and "[MISSING_PAGE" not in text:
                            out[idx] = text
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Nougat failed on page {idx + 1}: {exc}")
        return out


def get_extractor() -> Optional[PageMarkdownExtractor]:
    """Extractor configured by ``STEM_EXTRACTION_BACKEND``; None = keep plain text extraction."""
    backend = (settings.stem_extraction_backend or "none").strip().lower()
    if backend in ("", "none", "off", "false"):
        return None
    if backend == "vlm":
        return VLMExtractor()
    if backend == "marker":
        if MarkerExtractor.available():
            return MarkerExtractor()
        logger.warning("STEM_EXTRACTION_BACKEND=marker but `marker-pdf` is not installed — using plain text.")
        return None
    if backend == "nougat":
        extractor = NougatExtractor()
        if extractor.available():
            return extractor
        logger.warning(f"STEM_EXTRACTION_BACKEND=nougat but '{settings.nougat_cmd}' is not on PATH — using plain text.")
        return None
    logger.warning(f"Unknown STEM_EXTRACTION_BACKEND='{backend}' — using plain text.")
    return None


def extract_formula_pages(file_path: str, n_pages: int,
                          extractor: Optional[PageMarkdownExtractor] = None) -> Dict[int, str]:
    """Markdown/LaTeX for the math pages of a PDF (``{}`` when disabled or on any failure)."""
    extractor = extractor if extractor is not None else get_extractor()
    if extractor is None or n_pages <= 0:
        return {}
    try:
        targets = list(range(n_pages)) if settings.stem_extract_all_pages else detect_math_pages(file_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Math page detection failed: {exc}")
        return {}
    cap = max(0, settings.vlm_max_pages)
    if cap and len(targets) > cap:
        logger.info(f"{len(targets)} math pages found; converting only the first {cap} (VLM_MAX_PAGES).")
        targets = targets[:cap]
    if not targets:
        return {}
    try:
        result = extractor.extract(file_path, targets)
    except Exception as exc:  # noqa: BLE001 - whole-backend failure => plain text for every page
        logger.warning(f"Formula extraction backend '{getattr(extractor, 'name', '?')}' failed: {exc}")
        return {}
    return {i: t for i, t in result.items() if 0 <= i < n_pages and t and t.strip()}
