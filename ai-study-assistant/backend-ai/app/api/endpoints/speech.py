import logging
import threading
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.agent.tools import language_tool
from app.api.dependencies import get_current_user
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core.concurrency import too_busy, user_request
from app.models.models import User
from app.speech import analysis, transcriber

logger = logging.getLogger(__name__)
router = APIRouter()

_STT_SLOTS = threading.BoundedSemaphore(2)     # Whisper is CPU/GPU heavy: at most 2 transcriptions at a time
_LANG_OK = {"auto", "en", "vi", "fr", "de", "es", "ja", "ko", "zh"}


def _looks_like_audio(head: bytes) -> bool:
    """Magic bytes for wav / mp3 / ogg / flac / webm(matroska) / mp4-m4a."""
    return (
        (head[:4] == b"RIFF" and head[8:12] == b"WAVE") or head[:3] == b"ID3" or
        (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0) or head[:4] == b"OggS" or
        head[:4] == b"fLaC" or head[:4] == b"\x1aE\xdf\xa3" or head[4:8] == b"ftyp"
    )


async def _read_audio(file: UploadFile) -> bytes:
    limit = settings.stt_max_audio_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File âm thanh quá lớn (tối đa {settings.stt_max_audio_mb} MB).")
    if not data or not _looks_like_audio(data[:16]):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Định dạng âm thanh không được hỗ trợ (dùng wav, mp3, ogg, webm, m4a hoặc flac).")
    return data


def _check_language(language: Optional[str]) -> Optional[str]:
    lang = (language or "auto").lower()
    if lang not in _LANG_OK:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Ngôn ngữ không hỗ trợ: {language}.")
    return lang


def _transcribe_guarded(audio: bytes, language: str) -> transcriber.Transcript:
    if not _STT_SLOTS.acquire(timeout=2.0):
        raise too_busy("Hệ thống đang xử lý nhiều bản ghi. Vui lòng thử lại sau vài giây.")
    try:
        return transcriber.transcribe_audio(audio, language)
    finally:
        _STT_SLOTS.release()


async def _run_stt(audio: bytes, language: str) -> transcriber.Transcript:
    try:
        return await run_in_threadpool(_transcribe_guarded, audio, language)
    except HTTPException:
        raise
    except transcriber.SpeechUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except transcriber.AudioTooLong as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - corrupt audio, decoder errors...
        logger.warning("Transcription failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Không đọc được bản ghi âm. Hãy thử ghi lại.") from exc


def _grammar(text: str, language: str):
    """Grammar suggestions through LanguageTool when configured (English only here); never raises."""
    if not (settings.language_tool_enabled and settings.languagetool_url and language.startswith("en") and text):
        return None
    try:
        res = language_tool._check_languagetool(text, "en-US")
    except (httpx.HTTPError, ValueError, OSError):
        return None
    return {"corrected": res.corrected, "suggestions": res.explanations}


@router.get("/status")
async def speech_status(current_user: User = Depends(get_current_user)):
    return {
        "enabled": settings.stt_enabled, "available": transcriber.is_available(),
        "model": settings.whisper_model, "max_seconds": settings.stt_max_audio_seconds,
        "max_mb": settings.stt_max_audio_mb, "languages": sorted(_LANG_OK),
    }


@router.post("/transcribe")
@limiter.limit("12/minute")
async def transcribe(
    request: Request, file: UploadFile = File(...), language: str = Form(default="auto"),
    current_user: User = Depends(get_current_user),
):
    lang = _check_language(language)
    audio = await _read_audio(file)
    with user_request(str(current_user.id)):
        t = await _run_stt(audio, lang)
    return {"text": t.text, "language": t.language, "duration_s": round(t.duration_s, 2), "segments": t.segments}


@router.post("/practice")
@limiter.limit("12/minute")
async def practice(
    request: Request, file: UploadFile = File(...), language: str = Form(default="auto"),
    reference_text: str = Form(default="", max_length=2000), current_user: User = Depends(get_current_user),
):
    """Transcribe a recording and return fluency metrics, WER against an optional reference, grammar, and tips."""
    lang = _check_language(language)
    audio = await _read_audio(file)
    with user_request(str(current_user.id)):
        t = await _run_stt(audio, lang)

    detected = (t.language or (lang if lang != "auto" else "en")).lower()
    metrics = analysis.fluency_metrics(t.words, t.duration_s, detected)
    ref = reference_text.strip()
    wer = analysis.wer_report(ref, t.text) if ref else None
    overall = analysis.overall_score(metrics, wer["accuracy"] if wer else None)
    grammar = await run_in_threadpool(_grammar, t.text, detected)
    return {
        "transcript": t.text, "language": detected, "duration_s": round(t.duration_s, 2),
        "metrics": metrics, "reference": wer, "overall_score": overall, "grammar": grammar,
        "tips": analysis.tips(metrics, wer),
    }
