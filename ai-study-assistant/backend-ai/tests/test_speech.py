"""Speaking practice: alignment/WER, fluency metrics, scoring, and the API (Whisper is faked)."""
import pytest

from app.api.endpoints import speech as speech_ep
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.speech import analysis, transcriber

PW = "firstpassword1"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64


async def _user(client, name):
    res = await client.post("/api/auth/register", json={"username": name, "email": f"{name}@example.com", "password": PW})
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def words(spec, start=0.0, step=0.4):
    """[(word, prob)] -> word timing dicts, evenly spaced."""
    out, t = [], start
    for w, p in spec:
        out.append({"word": w, "start": t, "end": t + step * 0.8, "prob": p})
        t += step
    return out


# ---- pure analysis -----------------------------------------------------------------------------------------------
def test_wer_counts_substitutions_deletions_insertions():
    r = analysis.wer_report("The quick brown fox", "the quick red fox jumps")
    assert (r["substitutions"], r["deletions"], r["insertions"]) == (1, 0, 1)
    assert r["wer"] == 0.5 and r["accuracy"] == 0.5
    assert r["mispronounced_or_wrong"] == [{"expected": "brown", "heard": "red"}]
    miss = analysis.wer_report("I like green tea", "I like tea")
    assert miss["deletions"] == 1 and miss["missed_words"] == ["green"]


def test_wer_edge_cases():
    assert analysis.wer_report("", "")["wer"] == 0.0
    assert analysis.wer_report("", "something")["wer"] == 1.0
    assert analysis.wer_report("hello", "")["wer"] == 1.0 and analysis.wer_report("hello", "")["accuracy"] == 0.0
    assert analysis.wer_report("Hello, World!", "hello world")["wer"] == 0.0           # punctuation/case ignored
    assert analysis.wer_report("Xin chào", "xin chao")["wer"] == 0.5                    # diacritics matter in Vietnamese


def test_fluency_metrics_pace_pauses_fillers_and_clarity():
    ws = words([("I", 0.9), ("um", 0.8), ("think", 0.95), ("this", 0.3), ("is", 0.9)] * 4, step=0.4)   # 20 words
    ws[10]["start"] += 1.5                                                                             # a long pause
    ws[10]["end"] += 1.5
    for w in ws[11:]:
        w["start"] += 1.5
        w["end"] += 1.5
    m = analysis.fluency_metrics(ws, 12.0, "en")
    assert m["word_count"] == 20 and m["long_pauses"] == 1 and m["fillers"] == 4
    assert 100 < m["wpm"] < 150 and m["clarity"] == round(100 * (0.9 + 0.8 + 0.95 + 0.3 + 0.9) / 5, 1)
    assert {w["word"] for w in m["low_confidence_words"]} == {"this"}


def test_pace_score_band_and_overall_weights():
    assert analysis._pace_score(130, (110, 160)) == 100 and analysis._pace_score(60, (110, 160)) == 0.0
    assert analysis._pace_score(185, (110, 160)) == 100 - 2.5 * 25
    m = {"clarity": 80.0, "pace_score": 100.0, "pause_score": 60.0}
    assert analysis.overall_score(m) == round(0.45 * 80 + 0.3 * 100 + 0.25 * 60, 1)
    assert analysis.overall_score(m, accuracy=1.0) == round(0.4 * 100 + 0.25 * 80 + 0.2 * 100 + 0.15 * 60, 1)
    assert analysis.overall_score({"clarity": None, "pace_score": 50.0, "pause_score": None}) == 50.0   # renormalised
    assert analysis.overall_score({"clarity": None, "pace_score": None, "pause_score": None}) is None


def test_silence_and_tips():
    m = analysis.fluency_metrics([], 3.0, "en")
    assert m["word_count"] == 0 and m["wpm"] == 0 and m["pace_score"] is None
    assert "micro" in analysis.tips(m)[0]
    slow = analysis.fluency_metrics(words([("hello", 0.9)] * 6, step=1.5), 10, "en")
    assert any("chậm" in t for t in analysis.tips(slow))


# ---- API ---------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)


def fake_transcript(text="I like green tea", ws=None):
    ws = ws or words([("I", .9), ("like", .9), ("green", .4), ("tea", .95)])
    return transcriber.Transcript(text=text, language="en", duration_s=2.0,
                                  segments=[{"start": 0.0, "end": 1.6, "text": text}], words=ws)


async def _post(client, headers, path, audio=WAV, **data):
    return await client.post(f"/api/speech/{path}", headers=headers, data=data,
                             files={"file": ("rec.wav", audio, "audio/wav")})


async def test_practice_returns_metrics_reference_alignment_and_tips(app_client, monkeypatch):
    monkeypatch.setattr(transcriber, "transcribe_audio", lambda audio, language: fake_transcript())
    h = await _user(app_client, "sp1")
    res = await _post(app_client, h, "practice", reference_text="I like green tea", language="en")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["transcript"] == "I like green tea" and body["reference"]["accuracy"] == 1.0
    assert body["metrics"]["word_count"] == 4 and body["overall_score"] is not None
    assert {w["word"] for w in body["metrics"]["low_confidence_words"]} == {"green"} and body["tips"]


async def test_practice_without_reference_and_transcribe_endpoint(app_client, monkeypatch):
    monkeypatch.setattr(transcriber, "transcribe_audio", lambda audio, language: fake_transcript())
    h = await _user(app_client, "sp2")
    free = (await _post(app_client, h, "practice")).json()
    assert free["reference"] is None and free["overall_score"] is not None
    plain = (await _post(app_client, h, "transcribe", language="auto")).json()
    assert plain["text"] == "I like green tea" and plain["segments"][0]["start"] == 0.0


async def test_rejects_bad_audio_large_files_and_bad_language(app_client, monkeypatch):
    monkeypatch.setattr(transcriber, "transcribe_audio", lambda audio, language: fake_transcript())
    h = await _user(app_client, "sp3")
    assert (await _post(app_client, h, "transcribe", audio=b"%PDF-1.4 not audio")).status_code == 415
    assert (await _post(app_client, h, "transcribe", audio=b"")).status_code == 415
    assert (await _post(app_client, h, "transcribe", language="klingon")).status_code == 400
    monkeypatch.setattr(settings, "stt_max_audio_mb", 1)
    assert (await _post(app_client, h, "transcribe", audio=WAV + b"\x00" * (1024 * 1024 + 8))).status_code == 413


async def test_unavailable_too_long_and_decoder_errors_map_to_http_codes(app_client, monkeypatch):
    h = await _user(app_client, "sp4")

    def raising(exc):
        def fn(audio, language):
            raise exc
        return fn

    monkeypatch.setattr(transcriber, "transcribe_audio", raising(transcriber.SpeechUnavailable("chưa cài whisper")))
    res = await _post(app_client, h, "transcribe")
    assert res.status_code == 503 and "whisper" in res.json()["detail"]
    monkeypatch.setattr(transcriber, "transcribe_audio", raising(transcriber.AudioTooLong("quá dài")))
    assert (await _post(app_client, h, "transcribe")).status_code == 413
    monkeypatch.setattr(transcriber, "transcribe_audio", raising(RuntimeError("corrupt")))
    assert (await _post(app_client, h, "transcribe")).status_code == 422


async def test_busy_and_auth_and_status(app_client, monkeypatch):
    class Full:
        def acquire(self, timeout=None):
            return False

        def release(self):
            raise AssertionError("must not release a slot that was never acquired")

    monkeypatch.setattr(speech_ep, "_STT_SLOTS", Full())
    h = await _user(app_client, "sp5")
    assert (await _post(app_client, h, "transcribe")).status_code == 429
    assert (await app_client.post("/api/speech/transcribe", files={"file": ("a.wav", WAV, "audio/wav")})).status_code == 401
    st = (await app_client.get("/api/speech/status", headers=h)).json()
    assert st["enabled"] is True and "vi" in st["languages"] and st["max_seconds"] == settings.stt_max_audio_seconds


def test_transcriber_reports_disabled_and_missing_backend(monkeypatch):
    monkeypatch.setattr(settings, "stt_enabled", False)
    with pytest.raises(transcriber.SpeechUnavailable):
        transcriber.transcribe_audio(b"x")
    assert transcriber.is_available() is False
