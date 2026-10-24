"""Pure helpers for hybrid search (no Qdrant / model imports -> unit-testable).

* Tokenizer for Vietnamese + English that indexes every word twice: once with
  its original accents and once accent-folded (``~dieu``).  A query typed
  without accents ("he dieu hanh") therefore still matches "hệ điều hành",
  while an exact-accent match scores higher (both namespaces hit).
* BM25-style sparse vectors.  Term-frequency saturation + length
  normalisation are computed here; IDF is applied by Qdrant itself
  (``Modifier.IDF``) over the *whole* collection, so a keyword leg is a real
  retrieval step instead of re-scoring the dense candidates.
* Weighted Reciprocal Rank Fusion, min-max normalisation and de-duplication.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Hashable, List, Sequence, Tuple

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the
    this to was were will with what which who how why when where do does did
    và là của có các một những được trong cho với này để không khi thì mà như
    từ cũng đã sẽ bị tại về theo thế nào gì sao ra vào lại rất nên cần hay
    """.split()
)

# Bumping this constant (or changing the tokenizer) requires re-ingestion.
BM25_K1 = 1.2
BM25_B = 0.75
BM25_AVG_LEN = 128.0  # assumed average chunk length (tokens); IDF is server-side


def fold_diacritics(text: str) -> str:
    """'Hệ điều hành' -> 'He dieu hanh'."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _words(text: str) -> List[str]:
    text = unicodedata.normalize("NFC", text or "").lower()
    return [w for w in _WORD_RE.findall(text) if len(w) > 1 or w.isdigit()]


def tokenize(text: str, bigrams: bool = True) -> Tuple[List[str], int]:
    """Return ``(terms, n_words)``; ``terms`` may repeat (used for TF)."""
    words = _words(text)
    terms: List[str] = []
    for w in words:
        if w in _STOPWORDS:
            continue
        terms.append(w)
        terms.append("~" + fold_diacritics(w))
    if bigrams:
        for a, b in zip(words, words[1:]):
            if a in _STOPWORDS or b in _STOPWORDS:
                continue
            terms.append(f"{a}_{b}")
            terms.append(f"~{fold_diacritics(a)}_{fold_diacritics(b)}")
    return terms, len(words)


def term_index(term: str) -> int:
    """Stable (process-independent) 31-bit id for a term."""
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def sparse_document_vector(text: str, bigrams: bool = True) -> Tuple[List[int], List[float]]:
    """BM25 term weights (without IDF) for a chunk, as sparse (indices, values)."""
    terms, n_words = tokenize(text, bigrams)
    if not terms:
        return [], []
    norm = BM25_K1 * (1.0 - BM25_B + BM25_B * max(n_words, 1) / BM25_AVG_LEN)
    weights: Dict[int, float] = {}
    for term, tf in Counter(terms).items():
        idx = term_index(term)
        weights[idx] = weights.get(idx, 0.0) + tf * (BM25_K1 + 1.0) / (tf + norm)
    items = sorted(weights.items())
    return [i for i, _ in items], [float(v) for _, v in items]


def sparse_query_vector(text: str, bigrams: bool = True) -> Tuple[List[int], List[float]]:
    """Query side: every distinct term weighs 1.0 (Qdrant multiplies by IDF)."""
    terms, _ = tokenize(text, bigrams)
    indices = sorted({term_index(t) for t in terms})
    return indices, [1.0] * len(indices)


def weighted_rrf(
    ranked_lists: Sequence[Tuple[float, Sequence[Hashable]]], k: int = 60
) -> Dict[Hashable, float]:
    """Weighted Reciprocal Rank Fusion: sum(weight / (k + rank)), rank from 1."""
    scores: Dict[Hashable, float] = {}
    for weight, keys in ranked_lists:
        if weight <= 0:
            continue
        for rank, key in enumerate(keys, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return scores


def minmax(values: Sequence[float]) -> List[float]:
    """Scale to [0, 1]; a constant list maps to all 1.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _location(hit: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        hit.get("document_id"), hit.get("page_number"),
        hit.get("slide_number"), hit.get("section_title"),
    )


def drop_redundant_summaries(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove an extractive summary chunk when the detail chunk it was copied
    from (same page/slide/section) is also in the list; order is preserved."""
    kept: List[Dict[str, Any]] = []
    for i, hit in enumerate(hits):
        if hit.get("is_summary"):
            text = " ".join(hit["content"].split())
            redundant = any(
                not other.get("is_summary")
                and _location(other) == _location(hit)
                and text in " ".join(other["content"].split())
                for j, other in enumerate(hits) if j != i
            )
            if redundant:
                continue
        kept.append(hit)
    return kept
