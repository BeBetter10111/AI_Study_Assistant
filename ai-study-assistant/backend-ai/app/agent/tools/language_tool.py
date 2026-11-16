"""Grammar / wording feedback for language-learning questions.

Uses a LanguageTool HTTP server when ``LANGUAGETOOL_URL`` is set; otherwise a short
LLM correction prompt.  Fail-open: any error returns an empty result so chat still
answers via RAG.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urljoin

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_LANG_HINT = re.compile(
    r"(?:"
    r"\b(?:grammar|proofread|correct\s+my|fix\s+my|translate|pronounce|pronunciation|"
    r"spell\s*check|rewrite)\b|"
    r"ngữ pháp|sửa lỗi|chữa bài|dịch|phát âm|viết lại|chỉnh sửa câu"
    r")",
    re.IGNORECASE,
)

_TTS_HINT = re.compile(
    r"(?:\bpronounc|\btts\b|đọc (?:to|lại|giúp)|phát âm|hear\s+(?:it|this)|speak\s+(?:it|this))",
    re.IGNORECASE,
)

# Quoted or fenced learner text to correct.
_QUOTED = re.compile(r"[\"“”'](.+?)[\"“”']", re.S)
_FENCED = re.compile(r"```(?:\w+)?\s*([\s\S]+?)```")


@dataclass
class LanguageResult:
    ok: bool
    original: str = ""
    corrected: str = ""
    explanations: List[str] = field(default_factory=list)
    tts_text: str = ""          # text the client may speak (Web Speech / optional TTS)
    want_tts: bool = False
    error: str = ""

    def as_prompt_block(self) -> str:
        if not self.ok:
            return f"Language tool unavailable: {self.error}" if self.error else ""
        parts = [f"Original:\n{self.original}", f"Corrected:\n{self.corrected or self.original}"]
        if self.explanations:
            parts.append("Notes:\n" + "\n".join(f"- {e}" for e in self.explanations[:8]))
        return "\n\n".join(parts)


def looks_like_language_task(question: str) -> bool:
    return bool(_LANG_HINT.search(question or ""))


def wants_tts(question: str) -> bool:
    return bool(_TTS_HINT.search(question or ""))


def _extract_learner_text(question: str) -> str:
    q = (question or "").strip()
    m = _FENCED.search(q)
    if m:
        return m.group(1).strip()
    m = _QUOTED.search(q)
    if m:
        return m.group(1).strip()
    # Strip the instruction prefix if present.
    cleaned = _LANG_HINT.sub("", q).strip(" :,-")
    return cleaned or q


def _check_languagetool(text: str, language: str = "en-US") -> LanguageResult:
    base = (settings.languagetool_url or "").rstrip("/") + "/"
    url = urljoin(base, "v2/check")
    with httpx.Client(timeout=8.0) as client:
        resp = client.post(url, data={"text": text, "language": language})
        resp.raise_for_status()
        data = resp.json()
    corrected = text
    # Apply replacements from the end so offsets stay valid.
    matches = sorted(data.get("matches") or [], key=lambda m: m.get("offset", 0), reverse=True)
    explanations: List[str] = []
    for match in matches:
        offset = int(match.get("offset") or 0)
        length = int(match.get("length") or 0)
        reps = match.get("replacements") or []
        msg = match.get("message") or match.get("shortMessage") or "suggestion"
        if reps:
            replacement = reps[0].get("value", "")
            explanations.append(f"{msg} → «{replacement}»")
            corrected = corrected[:offset] + replacement + corrected[offset + length:]
        else:
            explanations.append(msg)
    explanations.reverse()
    return LanguageResult(
        ok=True,
        original=text,
        corrected=corrected,
        explanations=explanations[:12],
        tts_text=corrected or text,
    )


def _check_llm(text: str, question: str) -> LanguageResult:
    from app.core.llm_engine import llm_engine

    system = (
        "You are a language tutor. Correct the learner's text. "
        "Reply in this exact format:\n"
        "CORRECTED: <full corrected text>\n"
        "NOTES:\n- <short note>\n- <short note>\n"
        "Keep the learner's language. Do not add a preamble."
    )
    prompt = f"Learner request: {question}\n\nText to correct:\n{text}"
    raw = llm_engine.generate_rag_response(
        prompt=prompt, system_prompt=system, temperature=0.1, max_tokens=512,
    ).strip()
    corrected = text
    notes: List[str] = []
    if "CORRECTED:" in raw:
        body = raw.split("CORRECTED:", 1)[1]
        if "NOTES:" in body:
            corr, note_block = body.split("NOTES:", 1)
            corrected = corr.strip() or text
            notes = [
                ln.lstrip("-• ").strip()
                for ln in note_block.strip().splitlines()
                if ln.strip()
            ]
        else:
            corrected = body.strip() or text
    else:
        corrected = raw or text
    return LanguageResult(
        ok=True,
        original=text,
        corrected=corrected,
        explanations=notes[:12],
        tts_text=corrected or text,
    )


def try_correct(question: str) -> Optional[LanguageResult]:
    """Run grammar/pronunciation helper when the question looks like a language task."""
    if not settings.language_tool_enabled or not looks_like_language_task(question):
        return None
    text = _extract_learner_text(question)
    if not text or len(text) > 4000:
        return LanguageResult(ok=False, original=text, error="No text to correct")
    want = wants_tts(question)
    try:
        if settings.languagetool_url:
            result = _check_languagetool(text)
        else:
            result = _check_llm(text, question)
        result.want_tts = want
        if want and not result.tts_text:
            result.tts_text = result.corrected or result.original
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Language tool failed: %s", exc)
        return LanguageResult(ok=False, original=text, want_tts=want, error=str(exc),
                              tts_text=text if want else "")
