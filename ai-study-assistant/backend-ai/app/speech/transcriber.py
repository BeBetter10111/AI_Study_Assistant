from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class SpeechUnavailable(RuntimeError):
    """STT disabled or the whisper backend is not installed."""


class AudioTooLong(ValueError):
    """Recording longer than ``stt_max_audio_seconds``."""


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: float
    segments: List[Dict[str, Any]] = field(default_factory=list)
    words: List[Dict[str, Any]] = field(default_factory=list)      # {word, start, end, prob}


_model = None
_lock = threading.Lock()


def _load_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:  # pragma: no cover - depends on the environment
                    raise SpeechUnavailable(
                    ) from exc
                logger.info("Loading Whisper model %r (%s, %s) …", settings.whisper_model,
                            settings.whisper_device, settings.whisper_compute_type)
                _model = WhisperModel(settings.whisper_model, device=settings.whisper_device,
                                      compute_type=settings.whisper_compute_type)
    return _model


def is_available() -> bool:
    if not settings.stt_enabled:
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe_audio(audio: bytes, language: Optional[str] = None) -> Transcript:
    """Blocking transcription with word timestamps (call from a thread)."""
    if not settings.stt_enabled:
        raise SpeechUnavailable("Tính năng nhận dạng giọng nói đang tắt (STT_ENABLED=false).")
    model = _load_model()
    lang = None if not language or language == "auto" else language
    segments, info = model.transcribe(
        io.BytesIO(audio), language=lang, word_timestamps=True, vad_filter=True, beam_size=5,
    )
    if info.duration and info.duration > settings.stt_max_audio_seconds:
        raise AudioTooLong(f"Bản ghi dài {info.duration:.0f}s, tối đa {settings.stt_max_audio_seconds}s.")
    segs: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    for seg in segments:                       # generator: decoding happens here
        segs.append({"start": round(float(seg.start), 2), "end": round(float(seg.end), 2), "text": seg.text.strip()})
        for w in (seg.words or []):
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end),
                          "prob": float(w.probability)})
    text = " ".join(s["text"] for s in segs).strip()
    return Transcript(text=text, language=info.language or (lang or ""), duration_s=float(info.duration or 0.0),
                      segments=segs, words=words)
