from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Content format of chunks produced from formula-aware extraction (Markdown + LaTeX).
MARKDOWN_LATEX = "markdown_latex"

# Math blocks that must never be cut in half by the splitter (order matters: longest first).
_MATH_BLOCK_RE = re.compile(
    r"\$\$.+?\$\$"                                   # $$ display $$
    r"|\\\[.+?\\\]"                                  # \[ display \]
    r"|\\begin\{([A-Za-z]+\*?)\}.+?\\end\{\1\}"        # \begin{env} ... \end{env}
    r"|(?<![\\$])\$(?!\$)[^$\n]{1,300}?(?<![\\$])\$(?!\$)",  # inline $...$
    re.DOTALL,
)
_PH_OPEN, _PH_CLOSE = "\ue000", "\ue001"
_PH_RE = re.compile(f"{_PH_OPEN}(\\d+){_PH_CLOSE}")


def protect_math(text: str) -> tuple[str, List[str]]:
    """Replace math blocks with short separator-free placeholders."""
    blocks: List[str] = []

    def _sub(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"{_PH_OPEN}{len(blocks) - 1}{_PH_CLOSE}"

    if "$" not in text and "\\" not in text:
        return text, blocks
    return _MATH_BLOCK_RE.sub(_sub, text), blocks


def restore_math(text: str, blocks: List[str]) -> str:
    if not blocks:
        return text
    return _PH_RE.sub(lambda m: blocks[int(m.group(1))], text)


_encoder = None
_encoder_failed = False


def count_tokens(text: str) -> int:
    """cl100k token count (same unit as the splitter); ~4 chars/token if tiktoken is missing."""
    global _encoder, _encoder_failed
    if not _encoder_failed and _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _encoder_failed = True
    if _encoder is not None:
        try:
            return len(_encoder.encode(text or ""))
        except Exception:  # noqa: BLE001
            pass
    return (len(text or "") + 3) // 4


@dataclass
class ChunkData:
    content: str
    page_number: Optional[int] = None      
    slide_number: Optional[int] = None     # PPTX slide (1-indexed)
    slide_title: Optional[str] = None      # PPTX slide title text
    section_title: Optional[str] = None    # DOCX heading / section label
    chunk_index: int = 0                   # position within the document
    extra: Dict[str, Any] = field(default_factory=dict)


def _make_splitter(max_tokens: int = 512, overlap: int = 50) -> RecursiveCharacterTextSplitter:
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=max_tokens,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except Exception as e:
        logger.warning(f"tiktoken not available, falling back to char splitter: {e}")
        return RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * 4,
            chunk_overlap=overlap * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


def _split_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Split a single text block into sub-chunks, filter noise.

    Math-aware: LaTeX blocks (``$$..$$``, ``\\[..\\]``, ``\\begin{env}..\\end{env}``, inline ``$..$``)
    are swapped for placeholders before splitting and restored afterwards, so a formula is never
    split across two chunks (a chunk may slightly exceed ``max_tokens`` to keep it whole).
    """
    if not text or not text.strip():
        return []
    protected, blocks = protect_math(text)
    try:
        splitter = _make_splitter(max_tokens, overlap)
        pieces = splitter.split_text(protected)
    except Exception as e:
        logger.error(f"Chunking error: {e}")
        # character fallback (placeholders may be cut here; restore_math then leaves them as-is)
        size = max(max_tokens * 4, 1)
        step = max(size - overlap * 4, 1)
        pieces = [protected[i: i + size] for i in range(0, len(protected), step)]
    restored = [restore_math(p, blocks).strip() for p in pieces]
    return [p for p in restored if len(p) > 15]


# Structured chunking per source type
def chunk_pdf_pages(
    pages: List[str],
    max_tokens: int = 512,
    overlap: int = 50,
    markdown_pages: Optional[Collection[int]] = None,
) -> List[ChunkData]:
    """One chunk group per PDF page; pages that are too long are split further.

    ``markdown_pages`` (0-based) are pages produced by formula-aware extraction; their chunks are
    tagged ``extra["content_format"] = "markdown_latex"``.
    """
    markdown_pages = set(markdown_pages or ())
    chunks: List[ChunkData] = []
    global_idx = 0
    for page_num, page_text in enumerate(pages, start=1):
        sub_texts = _split_text(page_text, max_tokens, overlap)
        extra = {"content_format": MARKDOWN_LATEX} if (page_num - 1) in markdown_pages else {}
        for sub in sub_texts:
            chunks.append(
                ChunkData(
                    content=sub,
                    page_number=page_num,
                    chunk_index=global_idx,
                    extra=dict(extra),
                )
            )
            global_idx += 1
    return chunks


def chunk_pptx_slides(
    slides: List[Dict[str, Any]],
    max_tokens: int = 512,
    overlap: int = 50,
) -> List[ChunkData]:
    """One chunk group per PPTX slide.  Each slide dict has keys:
      ``slide_number`` (int), ``title`` (str|None), ``text`` (str).
    """
    chunks: List[ChunkData] = []
    global_idx = 0
    for slide in slides:
        slide_num = slide.get("slide_number", 0)
        title = slide.get("title") or ""
        text = slide.get("text", "")
        sub_texts = _split_text(text, max_tokens, overlap)
        for sub in sub_texts:
            chunks.append(
                ChunkData(
                    content=sub,
                    slide_number=slide_num,
                    slide_title=title if title else None,
                    chunk_index=global_idx,
                )
            )
            global_idx += 1
    return chunks


def chunk_docx_sections(
    sections: List[Dict[str, Any]],
    max_tokens: int = 512,
    overlap: int = 50,
) -> List[ChunkData]:
    """One chunk group per DOCX section/heading.  Each section dict has keys:
      ``heading`` (str|None), ``text`` (str).
    """
    chunks: List[ChunkData] = []
    global_idx = 0
    for section in sections:
        heading = section.get("heading") or ""
        text = section.get("text", "")
        sub_texts = _split_text(text, max_tokens, overlap)
        for sub in sub_texts:
            chunks.append(
                ChunkData(
                    content=sub,
                    section_title=heading if heading else None,
                    chunk_index=global_idx,
                )
            )
            global_idx += 1
    return chunks


# Legacy plain-text entry-point (kept for backward compatibility)
def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[ChunkData]:
    """Chunk a flat string.  Returns ChunkData list (no structural metadata)."""
    if not text or not text.strip():
        logger.warning("chunk_text received empty text, returning [].")
        return []
    sub_texts = _split_text(text, max_tokens, overlap)
    return [ChunkData(content=t, chunk_index=i) for i, t in enumerate(sub_texts)]