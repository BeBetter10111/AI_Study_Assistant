"""Hierarchical parser + chunker for legal / administrative / regulatory documents.

Vietnamese legal documents (Luật, Nghị định, Thông tư, Quyết định, Quy chế ...) follow a fixed
hierarchy::

    Phần  >  Chương  >  Mục  >  Điều  >  Khoản (1., 2., ...)  >  Điểm (a), b), đ) ...)

English regulations use ``Part > Chapter > Section > Article``.  Cutting such a document every
512 tokens regularly separates an article's conditions / exceptions ("trừ trường hợp quy định tại
khoản 3 ...") from the rule they qualify, so RAG answers miss them.  Here the *article* (Điều) is
the unit:

* an article that fits in the token budget  -> exactly one chunk;
* a longer article                           -> one chunk per clause (Khoản), each prefixed with
  the context header ``Chương II. ... > Mục 1 > Điều 5. <title>`` and the article's lead-in
  sentence; only an oversized clause is split further (every piece repeats the header).

Every chunk carries ``extra = {"domain": "legal", "hierarchy_path", "article_key", "article",
"clause"}`` so retrieval can pull the sibling clauses of a retrieved article
(``VectorStoreManager.expand_legal_articles``).
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from app.rag.chunking import ChunkData, _split_text, count_tokens

# Heading patterns (anchored at the start of a line)
_NUM = r"(?:\d+|[IVXLCDM]+)"
_PART_RE = re.compile(
    r"^\s*(PHẦN|Phần|PART|Part)\s+((?:THỨ|thứ|Thứ)\s+[^\s.:\-–—]+|" + _NUM + r")\b\s*[.:\-–—]?\s*(.*)$"
)
_CHAPTER_RE = re.compile(r"^\s*(CHƯƠNG|Chương|CHAPTER|Chapter)\s+(" + _NUM + r")\b\s*[.:\-–—]?\s*(.*)$")
_SECTION_RE = re.compile(r"^\s*(MỤC|Mục|SECTION|Section)\s+(" + _NUM + r")\b\s*[.:\-–—]?\s*(.*)$")
_ARTICLE_RE = re.compile(
    r"^\s*(ĐIỀU|Điều|ARTICLE|Article)\s+(\d+[a-zA-Z]?)\s*(?:[.:\-–—]\s*(.*)|$)"
)
_CLAUSE_RE = re.compile(r"^\s*(\d+)\.\s+\S")
_APPENDIX_RE = re.compile(r"^\s*(PHỤ LỤC|Phụ lục|APPENDIX|Appendix|ANNEX|Annex)\b(.*)$")
# Any heading (used to decide whether the next line can be a chapter/section title)
_ANY_HEADING = (_PART_RE, _CHAPTER_RE, _SECTION_RE, _ARTICLE_RE, _APPENDIX_RE)


@dataclass
class LegalClause:
    number: str
    text: str


@dataclass
class LegalArticle:
    number: str
    title: str
    label: str                       # "Điều" | "Article"
    hierarchy: List[str] = field(default_factory=list)   # ["Phần thứ nhất ...", "Chương II. ...", "Mục 1. ..."]
    lead: str = ""                   # text before the first clause
    clauses: List[LegalClause] = field(default_factory=list)
    page_number: Optional[int] = None
    seq: int = 0                     # position of the article in the document

    @property
    def heading(self) -> str:
        return f"{self.label} {self.number}. {self.title}".rstrip(". ").strip() if self.title \
            else f"{self.label} {self.number}"

    @property
    def hierarchy_path(self) -> str:
        return " > ".join([*self.hierarchy, self.heading])

    @property
    def body(self) -> str:
        parts = [self.lead] if self.lead else []
        parts += [c.text for c in self.clauses]
        return "\n".join(p for p in parts if p).strip()

    @property
    def key(self) -> str:
        """Document-local id (article numbers can repeat, e.g. in appendices)."""
        return f"{self.seq}:{self.number}"


@dataclass
class LegalDocument:
    preamble: str = ""
    preamble_page: Optional[int] = None
    articles: List[LegalArticle] = field(default_factory=list)
    appendices: List[Tuple[str, str, Optional[int]]] = field(default_factory=list)  # (title, text, page)


# Detection
def _article_numbers(text: str) -> List[int]:
    nums: List[int] = []
    for line in (text or "").splitlines():
        m = _ARTICLE_RE.match(line)
        if m:
            nums.append(int(re.match(r"\d+", m.group(2)).group(0)))
    return nums


def legal_structure_score(text: str) -> Tuple[int, float]:
    """(number of article headings, share of headings numbered in sequence)."""
    nums = _article_numbers(text)
    if len(nums) < 2:
        return len(nums), 0.0
    sequential = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
    return len(nums), sequential / (len(nums) - 1)


def is_legal_document(text: str, min_articles: int = 3, min_sequential: float = 0.6) -> bool:
    count, ratio = legal_structure_score(text)
    return count >= max(2, min_articles) and ratio >= min_sequential


# Parsing
def _clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(" .:-–—\t")).strip()


def _is_heading(line: str) -> bool:
    return any(rx.match(line) for rx in _ANY_HEADING)


def _join_label(kind: str, number: str, title: str) -> str:
    base = f"{kind} {number}".strip()
    return f"{base}. {title}" if title else base


def parse_legal(text: str, page_offsets: Optional[Sequence[int]] = None) -> LegalDocument:
    """Parse a legal text into articles with their hierarchy.

    :param page_offsets: character offset where each page starts in ``text`` (page 1 first);
                         used to give every article the page on which it starts.
    """
    doc = LegalDocument()
    offsets = list(page_offsets or [])

    def page_of(pos: int) -> Optional[int]:
        if not offsets:
            return None
        return max(1, bisect.bisect_right(offsets, pos))

    lines = (text or "").splitlines(keepends=True)
    starts: List[int] = []
    pos = 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln)
    lines = [ln.rstrip("\r\n") for ln in lines]

    part = chapter = section = ""
    current: Optional[LegalArticle] = None
    appendix: Optional[list] = None          # [title, [lines], page]
    preamble: List[str] = []
    last_article_num = -1
    seq = 0

    def title_from_next(i: int, inline: str) -> Tuple[str, int]:
        """Chapter/section titles are often on the next line (in capitals)."""
        title = _clean_title(inline)
        if title or i + 1 >= len(lines):
            return title, i
        nxt = lines[i + 1].strip()
        if nxt and len(nxt) <= 200 and not _is_heading(nxt) and not _CLAUSE_RE.match(nxt):
            return _clean_title(nxt), i + 1
        return "", i

    def flush_appendix():
        nonlocal appendix
        if appendix is not None:
            body = "\n".join(appendix[1]).strip()
            if body or appendix[0]:
                doc.appendices.append((appendix[0], body, appendix[2]))
            appendix = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        m = _APPENDIX_RE.match(line)
        if m and len(stripped) <= 150 and (current is not None or doc.articles):
            flush_appendix()
            current = None
            appendix = [_clean_title(stripped), [], page_of(starts[i])]
            i += 1
            continue

        m = _PART_RE.match(line)
        if m and appendix is None:
            title, i = title_from_next(i, m.group(3))
            part = _join_label(m.group(1).capitalize(), m.group(2), title)
            chapter = section = ""
            current = None
            i += 1
            continue
        m = _CHAPTER_RE.match(line)
        if m and appendix is None:
            title, i = title_from_next(i, m.group(3))
            chapter = _join_label(m.group(1).capitalize(), m.group(2), title)
            section = ""
            current = None
            i += 1
            continue
        m = _SECTION_RE.match(line)
        if m and appendix is None:
            title, i = title_from_next(i, m.group(3))
            section = _join_label(m.group(1).capitalize(), m.group(2), title)
            current = None
            i += 1
            continue

        m = _ARTICLE_RE.match(line)
        if m and appendix is None:
            num = int(re.match(r"\d+", m.group(2)).group(0))
            # Article numbers only go up; a smaller number at line start is a wrapped reference.
            if num > last_article_num or (num == last_article_num and m.group(2)[-1:].isalpha()):
                last_article_num = num
                label = "Điều" if m.group(1).lower().startswith("đ") else "Article"
                current = LegalArticle(
                    number=m.group(2), title=_clean_title(m.group(3) or ""), label=label,
                    hierarchy=[h for h in (part, chapter, section) if h],
                    page_number=page_of(starts[i]), seq=seq,
                )
                seq += 1
                doc.articles.append(current)
                i += 1
                continue

        if appendix is not None:
            appendix[1].append(line)
        elif current is None:
            if not doc.articles:
                if stripped and doc.preamble_page is None:
                    doc.preamble_page = page_of(starts[i])
                preamble.append(line)
            # Text between a chapter heading and its first article: keep it with the preamble only
            # before the first article; afterwards it is usually a title fragment -> ignored.
        else:
            cm = _CLAUSE_RE.match(line)
            expected = int(current.clauses[-1].number) + 1 if current.clauses else 1
            if cm and int(cm.group(1)) == expected:
                current.clauses.append(LegalClause(number=cm.group(1), text=stripped))
            elif current.clauses:
                current.clauses[-1].text += "\n" + stripped if stripped else ""
            elif stripped:
                current.lead = f"{current.lead}\n{stripped}".strip()
        i += 1

    flush_appendix()
    doc.preamble = "\n".join(preamble).strip()
    for art in doc.articles:
        for c in art.clauses:
            c.text = c.text.strip()
    return doc


# Chunking
def _first_sentence(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    m = re.search(r"(?<=[.:;!?])\s", text)
    sentence = text[: m.start()] if m else text
    return sentence if len(sentence) <= limit else sentence[:limit].rstrip() + "…"


def _legal_extra(article: LegalArticle, clause: Optional[str]) -> dict:
    return {
        "domain": "legal",
        "hierarchy_path": article.hierarchy_path,
        "article_key": article.key,
        "article": article.number,
        "clause": clause,
    }


def chunk_legal_document(
    doc: LegalDocument,
    max_tokens: int = 512,
    overlap: int = 50,
    start_index: int = 0,
) -> List[ChunkData]:
    chunks: List[ChunkData] = []
    idx = start_index

    def add(content: str, *, page: Optional[int], section: Optional[str], extra: dict) -> None:
        nonlocal idx
        chunks.append(ChunkData(content=content, page_number=page, section_title=section,
                                chunk_index=idx, extra=extra))
        idx += 1

    if doc.preamble:
        for piece in _split_text(doc.preamble, max_tokens, overlap):
            add(piece, page=doc.preamble_page, section=None,
                extra={"domain": "legal", "hierarchy_path": "Preamble"})

    for art in doc.articles:
        header = art.hierarchy_path
        body = art.body
        whole = f"{header}\n{body}".strip()
        meta_section = art.heading
        if count_tokens(whole) <= max_tokens or (not art.clauses and not body):
            add(whole, page=art.page_number, section=meta_section, extra=_legal_extra(art, None))
            continue

        header_budget = max(64, max_tokens - count_tokens(header) - 16)
        if not art.clauses:
            for piece in _split_text(body, header_budget, overlap):
                add(f"{header}\n{piece}", page=art.page_number, section=meta_section,
                    extra=_legal_extra(art, None))
            continue

        lead_sentence = _first_sentence(art.lead)
        # A lead-in longer than its first sentence is content in its own right.
        if art.lead and len(" ".join(art.lead.split())) > len(lead_sentence) + 20:
            for piece in _split_text(art.lead, header_budget, overlap):
                add(f"{header}\n{piece}", page=art.page_number, section=meta_section,
                    extra=_legal_extra(art, None))
        prefix = f"{header}\n{lead_sentence}" if lead_sentence else header
        clause_budget = max(64, max_tokens - count_tokens(prefix) - 16)
        for clause in art.clauses:
            if count_tokens(clause.text) <= clause_budget:
                pieces = [clause.text]
            else:
                pieces = _split_text(clause.text, clause_budget, overlap) or [clause.text]
            for piece in pieces:
                add(f"{prefix}\n{piece}", page=art.page_number, section=meta_section,
                    extra=_legal_extra(art, clause.number))

    for title, text, page in doc.appendices:
        for piece in _split_text(text, max_tokens, overlap) or ([title] if title else []):
            add(piece, page=page, section=title or None,
                extra={"domain": "legal", "hierarchy_path": title or "Appendix"})
    return chunks


def chunk_legal_text(
    text: str,
    page_offsets: Optional[Sequence[int]] = None,
    max_tokens: int = 512,
    overlap: int = 50,
) -> List[ChunkData]:
    """Parse + chunk in one call (returns [] when no article was found)."""
    doc = parse_legal(text, page_offsets)
    if not doc.articles:
        return []
    return chunk_legal_document(doc, max_tokens=max_tokens, overlap=overlap)
