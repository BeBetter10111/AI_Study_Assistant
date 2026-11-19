"""Speaking-practice analysis (pure functions, no model needed): alignment, WER and fluency metrics.

Inputs are the outputs of the speech-to-text model: words with start/end times and the model's confidence.

Metrics
-------
* **WER** (against a reference text):  WER = (S + D + I) / N  via word-level Levenshtein alignment
  (S substitutions, D deletions, I insertions, N reference words).  Accuracy = max(0, 1 - WER).
* **Pace**: words per minute over the speaking span (first word start -> last word end).
* **Pauses**: gaps between consecutive words >= ``LONG_PAUSE_S`` seconds.
* **Clarity** (pronunciation proxy): mean per-word recognition probability x 100. Whisper is not a pronunciation
  assessment model: a low value means *the recogniser was unsure*, which correlates with unclear or accented
  pronunciation but is **not** a phoneme-level score.
* **Overall**: weighted mean of the available components (see ``overall_score``).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

LONG_PAUSE_S = 0.7
LOW_CONFIDENCE = 0.5
PACE_BAND = {"en": (110.0, 160.0), "vi": (140.0, 190.0)}   # comfortable words-per-minute range
DEFAULT_BAND = (110.0, 170.0)

_FILLERS = {
    "en": {"um", "uh", "er", "erm", "hmm", "ah"},
    "vi": {"ờ", "à", "ừm", "ừ", "ư", "hừm", "ơ"},
}
_FILLER_PHRASES = {"en": ["you know", "i mean", "kind of", "sort of"], "vi": ["kiểu như", "thì là", "ý là"]}
_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_words(text: str) -> List[str]:
    """Lower-case, strip punctuation, collapse whitespace; keeps diacritics (they matter in Vietnamese)."""
    text = unicodedata.normalize("NFC", text or "").lower()
    return _PUNCT_RE.sub(" ", text).split()


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> List[Dict[str, Optional[str]]]:
    """Word-level Levenshtein alignment. Ops: equal | sub | del (reference word missing) | ins (extra word said)."""
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    ops: List[Dict[str, Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1):
            kind = "equal" if reference[i - 1] == hypothesis[j - 1] else "sub"
            ops.append({"type": kind, "ref": reference[i - 1], "hyp": hypothesis[j - 1]})
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append({"type": "del", "ref": reference[i - 1], "hyp": None})
            i -= 1
        else:
            ops.append({"type": "ins", "ref": None, "hyp": hypothesis[j - 1]})
            j -= 1
    ops.reverse()
    return ops


def wer_report(reference_text: str, transcript_text: str) -> Dict[str, Any]:
    ref, hyp = normalize_words(reference_text), normalize_words(transcript_text)
    ops = align(ref, hyp)
    s = sum(o["type"] == "sub" for o in ops)
    d = sum(o["type"] == "del" for o in ops)
    i = sum(o["type"] == "ins" for o in ops)
    n = len(ref)
    wer = (s + d + i) / n if n else (1.0 if hyp else 0.0)
    return {
        "wer": round(wer, 3), "accuracy": round(max(0.0, 1.0 - wer), 3),
        "substitutions": s, "deletions": d, "insertions": i, "reference_words": n,
        "alignment": ops,
        "mispronounced_or_wrong": [{"expected": o["ref"], "heard": o["hyp"]} for o in ops if o["type"] == "sub"][:15],
        "missed_words": [o["ref"] for o in ops if o["type"] == "del"][:15],
    }


def _pace_score(wpm: float, band) -> float:
    lo, hi = band
    dist = lo - wpm if wpm < lo else wpm - hi if wpm > hi else 0.0
    return max(0.0, 100.0 - 2.5 * dist)


def fluency_metrics(words: Sequence[Dict[str, Any]], duration_s: float, language: str = "en") -> Dict[str, Any]:
    """Pace, pauses, fillers and clarity from word timings. ``words``: [{word, start, end, prob}]."""
    lang = language if language in PACE_BAND else "en"
    band = PACE_BAND.get(lang, DEFAULT_BAND)
    tokens = [w for w in words if normalize_words(w.get("word", ""))]
    n_words = len(tokens)
    if n_words:
        span = max(float(tokens[-1]["end"]) - float(tokens[0]["start"]), 0.5)
    else:
        span = max(float(duration_s or 0.0), 0.5)
    wpm = n_words / (span / 60.0) if n_words else 0.0

    pauses = []
    for prev, cur in zip(tokens, tokens[1:]):
        gap = float(cur["start"]) - float(prev["end"])
        if gap >= LONG_PAUSE_S:
            pauses.append(round(gap, 2))
    long_pause_rate = len(pauses) / (span / 60.0) if n_words else 0.0

    text = " ".join(normalize_words(w["word"])[0] for w in tokens if normalize_words(w["word"]))
    fillers = sum(1 for w in text.split() if w in _FILLERS.get(lang, set()))
    fillers += sum(text.count(p) for p in _FILLER_PHRASES.get(lang, []))

    probs = [float(w["prob"]) for w in tokens if w.get("prob") is not None]
    clarity = round(100.0 * sum(probs) / len(probs), 1) if probs else None
    low = [{"word": w["word"].strip(), "confidence": round(float(w["prob"]), 2)}
           for w in tokens if w.get("prob") is not None and float(w["prob"]) < LOW_CONFIDENCE][:10]

    return {
        "word_count": n_words, "speaking_seconds": round(span, 1), "wpm": round(wpm, 1),
        "pace_band": list(band), "pace_score": round(_pace_score(wpm, band), 1) if n_words else None,
        "long_pauses": len(pauses), "longest_pause_s": max(pauses) if pauses else 0.0,
        "pause_score": round(max(0.0, 100.0 - 12.0 * long_pause_rate), 1) if n_words else None,
        "fillers": fillers, "clarity": clarity, "low_confidence_words": low,
    }


def overall_score(metrics: Dict[str, Any], accuracy: Optional[float] = None) -> Optional[float]:
    """Weighted mean of the available components (weights are renormalised when a component is missing).

    with reference: 0.40 accuracy + 0.25 clarity + 0.20 pace + 0.15 pauses
    free speech   :                0.45 clarity + 0.30 pace + 0.25 pauses
    """
    parts = [
        (0.40 if accuracy is not None else 0.0, None if accuracy is None else accuracy * 100.0),
        (0.25 if accuracy is not None else 0.45, metrics.get("clarity")),
        (0.20 if accuracy is not None else 0.30, metrics.get("pace_score")),
        (0.15 if accuracy is not None else 0.25, metrics.get("pause_score")),
    ]
    live = [(w, v) for w, v in parts if v is not None and w > 0]
    if not live:
        return None
    return round(sum(w * v for w, v in live) / sum(w for w, _ in live), 1)


def tips(metrics: Dict[str, Any], wer: Optional[Dict[str, Any]] = None) -> List[str]:
    """Short, rule-based feedback in Vietnamese."""
    out: List[str] = []
    if not metrics.get("word_count"):
        return ["Không nghe rõ nội dung. Hãy nói gần micro hơn và thử lại ở nơi yên tĩnh."]
    lo, hi = metrics["pace_band"]
    if metrics["wpm"] < lo:
        out.append(f"Tốc độ nói {metrics['wpm']:.0f} từ/phút, hơi chậm (khoảng thoải mái {lo:.0f}–{hi:.0f}). Thử nối ý liền mạch hơn.")
    elif metrics["wpm"] > hi:
        out.append(f"Tốc độ nói {metrics['wpm']:.0f} từ/phút, hơi nhanh (khoảng thoải mái {lo:.0f}–{hi:.0f}). Hãy nói chậm lại và nhấn vào từ khoá.")
    else:
        out.append(f"Tốc độ nói {metrics['wpm']:.0f} từ/phút nằm trong khoảng thoải mái.")
    if metrics["long_pauses"]:
        out.append(f"Có {metrics['long_pauses']} lần ngắt dài (dài nhất {metrics['longest_pause_s']:.1f}s). Chuẩn bị ý trước để nói liền mạch hơn.")
    if metrics["fillers"] >= 3:
        out.append(f"Dùng {metrics['fillers']} từ đệm (um/uh/ờ...). Thử dừng ngắn im lặng thay vì chèn từ đệm.")
    if metrics["low_confidence_words"]:
        words = ", ".join(w["word"] for w in metrics["low_confidence_words"][:5])
        out.append(f"Các từ máy nhận dạng kém chắc chắn (có thể phát âm chưa rõ): {words}. Hãy luyện lại các từ này.")
    if wer is not None:
        if wer["missed_words"]:
            out.append("Bỏ sót từ: " + ", ".join(wer["missed_words"][:6]) + ".")
        if wer["mispronounced_or_wrong"]:
            pairs = ", ".join(f"'{m['expected']}' → '{m['heard']}'" for m in wer["mispronounced_or_wrong"][:5])
            out.append("Từ bị nhận khác với mẫu: " + pairs + ".")
        if wer["accuracy"] >= 0.9:
            out.append("Độ chính xác so với câu mẫu rất tốt.")
    return out
