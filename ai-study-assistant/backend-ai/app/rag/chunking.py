"""Structured chunking with rich metadata (page number, slide title, section).

Each chunk is a dict rather than a bare string so that downstream components
(vector_store, chat endpoint) can surface precise citations like "Slide 12" or
"Page 3".  The plain-text fallback still returns ChunkData objects so the rest
of the pipeline never has to branch on type.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    content: str
    page_number: Optional[int] = None      
    slide_number: Optional[int] = None     # PPTX slide (1-indexed)
    slide_title: Optional[str] = None      # PPTX slide title text
    section_title: Optional[str] = None    # DOCX heading / section label
    chunk_index: int = 0                   # position within the document
    extra: Dict[str, Any] = field(default_factory=dict)


# Low-level text splitter (shared)
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
    """Split a single text block into sub-chunks, filter noise."""
    if not text or not text.strip():
        return []
    try:
        splitter = _make_splitter(max_tokens, overlap)
        pieces = splitter.split_text(text)
        return [p.strip() for p in pieces if len(p.strip()) > 15]
    except Exception as e:
        logger.error(f"Chunking error: {e}")
        # character fallback
        size = max(max_tokens * 4, 1)
        step = max(size - overlap * 4, 1)
        return [text[i: i + size] for i in range(0, len(text), step)]


# Structured chunking per source type
def chunk_pdf_pages(
    pages: List[str],
    max_tokens: int = 512,
    overlap: int = 50,
) -> List[ChunkData]:
    """One chunk group per PDF page; pages that are too long are split further."""
    chunks: List[ChunkData] = []
    global_idx = 0
    for page_num, page_text in enumerate(pages, start=1):
        sub_texts = _split_text(page_text, max_tokens, overlap)
        for sub in sub_texts:
            chunks.append(
                ChunkData(
                    content=sub,
                    page_number=page_num,
                    chunk_index=global_idx,
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