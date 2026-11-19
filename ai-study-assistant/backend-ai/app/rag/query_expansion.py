from __future__ import annotations

import csv
import glob
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import httpx

from app.config import settings
from app.rag.hybrid import fold_diacritics

logger = logging.getLogger(__name__)

DEFAULT_LEXICON = os.path.join(os.path.dirname(__file__), "lexicons", "medical.json")
# Every bundled domain lexicon (medical first, then science, law/economics, ...).
DEFAULT_LEXICONS = [DEFAULT_LEXICON] + sorted(
    p for p in glob.glob(os.path.join(os.path.dirname(__file__), "lexicons", "*.json")) if p != DEFAULT_LEXICON
)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_NGRAM = 6


def _norm_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(fold_diacritics(text or "").lower())


@dataclass
class Link:
    surface: str                 # text as written in the question
    canonical: str
    synonyms: List[str]
    type: str = ""
    id: Optional[str] = None
    source: str = "lexicon"

    def as_dict(self) -> Dict[str, Any]:
        return {"surface": self.surface, "canonical": self.canonical, "synonyms": self.synonyms,
                "type": self.type, "id": self.id, "source": self.source}


@dataclass
class Expansion:
    terms: List[str] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.links)

    def summary(self) -> str:
        """Human-readable, e.g. ``Panadol -> paracetamol, acetaminophen``."""
        parts = []
        for link in self.links:
            targets = [link.canonical] + [s for s in link.synonyms if s.lower() != link.surface.lower()][:2]
            targets = [t for t in targets if t.lower() != link.surface.lower()]
            parts.append(f"{link.surface} -> {', '.join(dict.fromkeys(targets))}" if targets else link.surface)
        return "; ".join(parts)


class TermLinker(Protocol):
    def link(self, text: str) -> List[Link]:
        ...


# Lexicon linker
def load_lexicon_entries(path: str) -> List[Dict[str, Any]]:
    """Read a JSON (``{"entries": [...]}`` or a list) or CSV lexicon."""
    if path.lower().endswith(".csv"):
        entries = []
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                canonical = (row.get("canonical") or "").strip()
                if not canonical:
                    continue
                entries.append({
                    "canonical": canonical,
                    "synonyms": [s.strip() for s in (row.get("synonyms") or "").split("|") if s.strip()],
                    "type": (row.get("type") or "").strip(),
                    "id": (row.get("id") or "").strip() or None,
                })
        return entries
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("entries", []) if isinstance(data, dict) else list(data)


class LexiconLinker:
    def __init__(self, entries: Sequence[Dict[str, Any]], source: str = "lexicon"):
        self.source = source
        self._entries: List[Dict[str, Any]] = []
        self._index: Dict[Tuple[str, ...], int] = {}
        for entry in entries:
            canonical = (entry.get("canonical") or "").strip()
            if not canonical:
                continue
            idx = len(self._entries)
            self._entries.append(entry)
            for surface in [canonical, *entry.get("synonyms", [])]:
                key = tuple(_norm_tokens(surface))
                if not key or len("".join(key)) < 3 or len(key) > _MAX_NGRAM:
                    continue
                self._index.setdefault(key, idx)

    @classmethod
    def from_paths(cls, paths: Sequence[str]) -> "LexiconLinker":
        entries: List[Dict[str, Any]] = []
        for path in paths:
            try:
                entries.extend(load_lexicon_entries(path))
            except Exception as exc:  # noqa: BLE001 - a broken extra lexicon must not break chat
                logger.warning(f"Could not load lexicon {path}: {exc}")
        return cls(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def link(self, text: str) -> List[Link]:
        # Keep original-cased words alongside the normalised ones to report the surface form.
        originals = _WORD_RE.findall(text or "")
        tokens = [fold_diacritics(w).lower() for w in originals]
        links: List[Link] = []
        used: set = set()
        i = 0
        while i < len(tokens):
            match = None
            for n in range(min(_MAX_NGRAM, len(tokens) - i), 0, -1):
                idx = self._index.get(tuple(tokens[i:i + n]))
                if idx is not None:
                    match = (n, idx)
                    break
            if match is None:
                i += 1
                continue
            n, idx = match
            if idx not in used:
                used.add(idx)
                entry = self._entries[idx]
                links.append(Link(
                    surface=" ".join(originals[i:i + n]),
                    canonical=entry["canonical"],
                    synonyms=list(entry.get("synonyms", [])),
                    type=entry.get("type", ""),
                    id=entry.get("id"),
                    source=self.source,
                ))
            i += n
        return links


# UMLS linker (optional)
_UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
_UMLS_STOP = frozenset("what which when where how why does with from that this there their about".split())


class UMLSLinker:
    def __init__(self, api_key: str, timeout_s: float = 2.0, max_candidates: int = 3):
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_candidates = max_candidates
        self._lookup = lru_cache(maxsize=512)(self._lookup_uncached)

    def _lookup_uncached(self, term: str) -> Optional[Tuple[str, str, Tuple[str, ...]]]:
        """(canonical name, CUI, synonyms) for an exact UMLS match, else None."""
        r = httpx.get(f"{_UMLS_BASE}/search/current", timeout=self.timeout_s,
                      params={"string": term, "searchType": "exact", "apiKey": self.api_key, "pageSize": 1})
        r.raise_for_status()
        results = r.json().get("result", {}).get("results", [])
        if not results or results[0].get("ui") in (None, "NONE"):
            return None
        cui, name = results[0]["ui"], results[0].get("name", term)
        synonyms: List[str] = []
        try:
            a = httpx.get(f"{_UMLS_BASE}/content/current/CUI/{cui}/atoms", timeout=self.timeout_s,
                          params={"language": "ENG", "apiKey": self.api_key, "pageSize": 15})
            a.raise_for_status()
            for atom in a.json().get("result", []):
                label = atom.get("name")
                if label and label.lower() not in {s.lower() for s in synonyms} and label.lower() != name.lower():
                    synonyms.append(label)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"UMLS atoms lookup failed for {cui}: {exc}")
        return name, cui, tuple(synonyms[:5])

    def candidates(self, text: str, skip: Sequence[str] = ()) -> List[str]:
        skip_norm = {fold_diacritics(s).lower() for s in skip}
        out: List[str] = []
        for word in _WORD_RE.findall(text or ""):
            norm = fold_diacritics(word).lower()
            if norm in skip_norm or norm in _UMLS_STOP or word.isdigit():
                continue
            if (word[:1].isupper() and len(word) >= 4) or len(word) >= 8:
                if word not in out:
                    out.append(word)
        return out[: self.max_candidates]

    def link(self, text: str, skip: Sequence[str] = ()) -> List[Link]:
        links: List[Link] = []
        for word in self.candidates(text, skip):
            try:
                found = self._lookup(word)
            except Exception as exc:  # noqa: BLE001 - fail-open
                logger.debug(f"UMLS lookup failed for '{word}': {exc}")
                continue
            if found:
                name, cui, synonyms = found
                links.append(Link(surface=word, canonical=name, synonyms=list(synonyms),
                                  type="umls", id=f"UMLS:{cui}", source="umls"))
        return links


# Public API
_default_linker: Optional[LexiconLinker] = None
_umls_linker: Optional[UMLSLinker] = None
_lock = threading.Lock()


def get_lexicon_linker() -> LexiconLinker:
    global _default_linker
    if _default_linker is None:
        with _lock:
            if _default_linker is None:
                extra = [p.strip() for p in (settings.medical_lexicon_paths or "").split(",") if p.strip()]
                _default_linker = LexiconLinker.from_paths([*DEFAULT_LEXICONS, *extra])
    return _default_linker


def _get_umls_linker() -> Optional[UMLSLinker]:
    global _umls_linker
    if not settings.umls_api_key:
        return None
    if _umls_linker is None:
        _umls_linker = UMLSLinker(settings.umls_api_key, timeout_s=settings.umls_timeout_s)
    return _umls_linker


def reset_linkers() -> None:
    """Drop cached linkers (tests / settings change)."""
    global _default_linker, _umls_linker
    _default_linker = None
    _umls_linker = None


def link_entities(query: str) -> List[Link]:
    links = get_lexicon_linker().link(query)
    umls = _get_umls_linker()
    if umls is not None:
        known = [w for link in links for w in link.surface.split()]
        links += umls.link(query, skip=known)
    return links


def expand_query(query: str, max_terms: Optional[int] = None) -> Expansion:
    """Link the entities of ``query`` and return the extra retrieval terms (fail-open)."""
    if not settings.query_expansion_enabled or not (query or "").strip():
        return Expansion()
    max_terms = settings.query_expansion_max_terms if max_terms is None else max_terms
    try:
        links = link_entities(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Query expansion failed: {exc}")
        return Expansion()
    query_norm = " ".join(_norm_tokens(query))
    terms: List[str] = []
    seen = set()
    # Round-robin: canonical names first, then synonyms, so every linked entity gets coverage.
    pools = [[link.canonical, *link.synonyms] for link in links]
    for depth in range(max((len(p) for p in pools), default=0)):
        for pool in pools:
            if depth >= len(pool) or len(terms) >= max_terms:
                continue
            term = pool[depth]
            norm = " ".join(_norm_tokens(term))
            if not norm or norm in seen or f" {norm} " in f" {query_norm} ":
                continue
            seen.add(norm)
            terms.append(term)
    return Expansion(terms=terms, links=links)
