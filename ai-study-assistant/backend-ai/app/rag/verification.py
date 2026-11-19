"""Post-generation verification: is every claim of the answer supported by the retrieved context?

LLMs can still invent facts even with a "answer only from the context" prompt. This module adds an
*independent*, cheap (no extra LLM call) check after generation and turns it into a confidence score
the UI can show next to the answer.

Pipeline
--------
1. Split the answer into **claims** (sentences) and remember which ``[n]`` citations each one carries.
2. Build **evidence windows** (1-2 sentences) from the retrieved chunks.
3. For each claim find the best window:  ``support = w_sem * cos + w_lex * lex_recall``
     * ``lex_recall`` = share of the claim's content words found in the window (accent-insensitive);
     * ``cos``        = cosine similarity of multilingual embeddings (skipped when unavailable -> lexical only).
4. A claim is **supported** iff ``support >= tau`` AND every number in the claim also appears in the context
   (numbers are the most common silent hallucination).
5. Confidence = ``0.6 * R + 0.4 * S - P`` where R is the length-weighted share of supported claims, S the
   length-weighted mean support and P a citation penalty (cited ``[n]`` that does not exist or whose
   passage does not support the claim).

Everything is fail-open: any error returns ``None`` and the answer is shown without a badge.
This is a heuristic *groundedness* estimate, not a proof of truth (see ``docs``/limitations).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from app.config import settings
from app.rag.hybrid import _STOPWORDS, _words, fold_diacritics

logger = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], np.ndarray]

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]+|\d{1,2}[.)])\s+")
_MARKUP_RE = re.compile(r"[*_`#>]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_ABSTAIN_RE = re.compile(
    r"(does not (?:contain|mention|provide|include|specify)|do not contain|no (?:relevant )?information|"
    r"not (?:mentioned|provided|found|covered|specified)|cannot (?:find|answer)|can't find|insufficient|"
    r"không (?:có|đề cập|tìm thấy|đủ|nêu|cung cấp|được nhắc)|chưa (?:có|đề cập)|không thể trả lời)",
    re.IGNORECASE,
)

MIN_CLAIM_TOKENS = 4
HIGH, MEDIUM = 0.75, 0.50


@dataclass
class ClaimCheck:
    text: str
    supported: bool
    score: float                      # best support score in [0, 1]
    evidence: Optional[int] = None    # 1-based index of the retrieved chunk that supports it best
    cited: List[int] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)
    weight: int = 1                   # number of content tokens (length weight)


@dataclass
class VerificationResult:
    confidence: Optional[float]       # None when not applicable (abstention / nothing to verify)
    label: str                        # high | medium | low | abstain | unverified
    supported_ratio: float = 0.0
    claims: List[ClaimCheck] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    used_embeddings: bool = False

    @property
    def unsupported_claims(self) -> List[str]:
        return [c.text for c in self.claims if not c.supported]

    def to_public(self, limit: int = 5) -> Dict[str, Any]:
        return {
            "confidence": None if self.confidence is None else round(self.confidence, 2),
            "confidence_label": self.label,
            "supported_ratio": round(self.supported_ratio, 2),
            "unsupported_claims": [t[:280] for t in self.unsupported_claims[:limit]],
            "invalid_citations": self.invalid_citations,
        }


# --------------------------------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------------------------------
def _content_tokens(text: str) -> List[str]:
    """Accent-folded content words (stopwords and 1-char tokens removed)."""
    out = []
    for w in _words(text.lower()):
        f = fold_diacritics(w)
        if len(f) > 1 and f not in _STOPWORDS and w not in _STOPWORDS:
            out.append(f)
    return out


def _numbers(text: str) -> List[str]:
    return [n.replace(",", ".").rstrip(".") for n in _NUM_RE.findall(text)]


def split_claims(answer: str) -> List[Dict[str, Any]]:
    """Sentences of the answer with their ``[n]`` citations; citations/markdown are removed from the text."""
    claims: List[Dict[str, Any]] = []
    for raw in _SENT_SPLIT.split(answer or ""):
        if not raw or not raw.strip():
            continue
        cited = [int(m) for m in _CITE_RE.findall(raw)]
        text = _CITE_RE.sub(" ", raw)
        text = _BULLET_RE.sub("", text)
        text = _MARKUP_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip(" -–—:;")
        if not text or text.endswith("?"):
            continue
        if len(_content_tokens(text)) < MIN_CLAIM_TOKENS:
            continue
        claims.append({"text": text, "cited": cited})
    return claims


def evidence_windows(hits: Sequence[Dict[str, Any]], max_windows: int = 400) -> List[Dict[str, Any]]:
    """Overlapping 2-sentence windows over each retrieved chunk (+ the chunk's 1-based index)."""
    windows: List[Dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        sents = [s.strip() for s in _SENT_SPLIT.split(hit.get("content") or "") if s and s.strip()]
        if not sents:
            continue
        for i in range(len(sents)):
            text = " ".join(sents[i:i + 2])
            windows.append({"text": text, "chunk": idx, "tokens": set(_content_tokens(text))})
            if len(windows) >= max_windows:
                return windows
    return windows


def _lex_recall(claim_tokens: Sequence[str], window_tokens: set) -> float:
    if not claim_tokens:
        return 0.0
    return sum(1 for t in claim_tokens if t in window_tokens) / len(claim_tokens)


def _default_embed(texts: Sequence[str]) -> np.ndarray:
    from app.rag import vector_store as vs

    return np.asarray(vs._get_embedding_model().encode(list(texts)), dtype=np.float32)


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


# --------------------------------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------------------------------
def verify_answer(
    answer: str,
    hits: Sequence[Dict[str, Any]],
    embed_fn: Optional[EmbedFn] = None,
    use_embeddings: Optional[bool] = None,
) -> Optional[VerificationResult]:
    """Return the verification result, or ``None`` when disabled / nothing to verify / on any error."""
    if not settings.verification_enabled or not answer or not hits:
        return None
    try:
        return _verify(answer, hits, embed_fn, use_embeddings)
    except Exception:  # noqa: BLE001 - verification must never break an answer
        logger.warning("Answer verification failed", exc_info=True)
        return None


def _verify(answer, hits, embed_fn, use_embeddings) -> Optional[VerificationResult]:
    if _ABSTAIN_RE.search(answer) and len(_content_tokens(answer)) <= 60:
        # "The context does not mention X" is an honest abstention, not a hallucination.
        return VerificationResult(confidence=None, label="abstain")

    claims = split_claims(answer)[: settings.verification_max_claims]
    if not claims:
        return VerificationResult(confidence=None, label="unverified")

    windows = evidence_windows(hits)
    if not windows:
        return VerificationResult(confidence=None, label="unverified")
    context_numbers = set()
    for hit in hits:
        context_numbers.update(_numbers(hit.get("content") or ""))

    # ---- lexical scores for every (claim, window); keep the best few windows per claim for the embedding step
    claim_tokens = [_content_tokens(c["text"]) for c in claims]
    lex = np.zeros((len(claims), len(windows)), dtype=np.float32)
    for i, toks in enumerate(claim_tokens):
        for j, w in enumerate(windows):
            lex[i, j] = _lex_recall(toks, w["tokens"])

    want_embed = settings.verification_use_embeddings if use_embeddings is None else use_embeddings
    sem = None
    if want_embed:
        try:
            fn = embed_fn or _default_embed
            keep = sorted({int(j) for row in lex for j in np.argsort(-row)[:3]})
            vecs = _unit(np.asarray(fn([c["text"] for c in claims] + [windows[j]["text"] for j in keep])))
            cvec, wvec = vecs[: len(claims)], vecs[len(claims):]
            sem = np.zeros_like(lex)
            sem[:, keep] = np.clip(cvec @ wvec.T, 0.0, 1.0)
        except Exception:  # noqa: BLE001 - fall back to lexical only
            logger.info("Verification embeddings unavailable; using lexical evidence only", exc_info=True)
            sem = None

    if sem is not None:
        combined, tau = 0.5 * sem + 0.5 * lex, settings.verification_threshold
    else:
        combined, tau = lex, settings.verification_lexical_threshold

    checks: List[ClaimCheck] = []
    per_chunk_best: List[Dict[int, float]] = []
    for i, c in enumerate(claims):
        j = int(np.argmax(combined[i]))
        score = float(combined[i, j])
        missing = [n for n in _numbers(c["text"]) if n not in context_numbers]
        chunk_scores: Dict[int, float] = {}
        for jj, w in enumerate(windows):
            chunk_scores[w["chunk"]] = max(chunk_scores.get(w["chunk"], 0.0), float(combined[i, jj]))
        per_chunk_best.append(chunk_scores)
        checks.append(ClaimCheck(
            text=c["text"], supported=(score >= tau and not missing), score=score,
            evidence=windows[j]["chunk"], cited=c["cited"], missing_numbers=missing,
            weight=max(1, len(claim_tokens[i])),
        ))

    total_w = sum(c.weight for c in checks)
    ratio = sum(c.weight for c in checks if c.supported) / total_w
    mean_score = sum(min(1.0, c.score / max(tau, 1e-6)) * c.weight for c in checks) / total_w

    n_chunks = len(hits)
    invalid = sorted({k for c in checks for k in c.cited if k < 1 or k > n_chunks})
    mismatched = 0
    for c, scores in zip(checks, per_chunk_best):
        valid = [k for k in c.cited if 1 <= k <= n_chunks]
        if valid and max(scores.get(k, 0.0) for k in valid) < 0.5 * tau:
            mismatched += 1
    penalty = min(0.3, 0.1 * len(invalid) + 0.1 * mismatched)

    confidence = max(0.0, min(1.0, 0.6 * ratio + 0.4 * min(1.0, mean_score) - penalty))
    label = "high" if confidence >= HIGH else "medium" if confidence >= MEDIUM else "low"
    return VerificationResult(
        confidence=confidence, label=label, supported_ratio=ratio, claims=checks,
        invalid_citations=invalid, used_embeddings=sem is not None,
    )
