import { useCallback, useEffect, useRef, useState } from "react";
import { getErrorMessage, getSpeechStatus, practiceSpeech } from "../../services/api";

const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
const pickMime = () =>
  typeof MediaRecorder === "undefined" ? "" : MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) || "";
const extFor = (mime) => (mime.includes("mp4") ? "m4a" : mime.includes("ogg") ? "ogg" : "webm");
const LANG_TTS = { auto: "en-US", en: "en-US", vi: "vi-VN" };

function Metric({ label, value, hint }) {
  return (
    <div className="metric" title={hint}>
      <div className="metric-value">{value ?? "—"}</div>
      <div className="field-hint">{label}</div>
    </div>
  );
}

// Colour-coded comparison between the reference sentence and what the recogniser heard.
function Alignment({ ops }) {
  return (
    <p className="align">
      {ops.map((o, i) => {
        if (o.type === "equal") return <span key={i} className="w w--ok">{o.hyp} </span>;
        if (o.type === "sub") return <span key={i} className="w w--bad" title={`Mẫu: ${o.ref}`}>{o.hyp} </span>;
        if (o.type === "del") return <span key={i} className="w w--miss" title="Bị bỏ sót">{o.ref} </span>;
        return <span key={i} className="w w--extra" title="Từ thừa">{o.hyp} </span>;
      })}
    </p>
  );
}

// Speaking practice: listen to a model sentence (browser TTS), record, get Whisper transcript + feedback.
export default function SpeakingView() {
  const [status, setStatus] = useState(null);
  const [language, setLanguage] = useState("en");
  const [reference, setReference] = useState("");
  const [rate, setRate] = useState(0.9);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    getSpeechStatus().then((res) => setStatus(res.data)).catch(() => setStatus({ enabled: false, available: false }));
  }, []);

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    clearInterval(timerRef.current);
  };

  useEffect(() => () => {
    stopStream();
    if (typeof window !== "undefined" && window.speechSynthesis) window.speechSynthesis.cancel();
  }, []);

  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  const speak = () => {
    const synth = window.speechSynthesis;
    if (!synth || !reference.trim()) return;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(reference);
    u.lang = LANG_TTS[language] || "en-US";
    u.rate = rate;
    synth.speak(u);
  };

  const upload = async (blob, mime) => {
    setBusy(true);
    try {
      const res = await practiceSpeech(blob, `recording.${extFor(mime)}`, language, reference.trim());
      setResult(res.data);
    } catch (err) {
      setError(getErrorMessage(err, "Không phân tích được bản ghi."));
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    setError(null);
    setResult(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("Trình duyệt không hỗ trợ ghi âm (cần HTTPS hoặc localhost).");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mime = pickMime();
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = () => {
        const type = recorder.mimeType || mime || "audio/webm";
        const blob = new Blob(chunksRef.current, { type });
        stopStream();
        setAudioUrl(URL.createObjectURL(blob));
        upload(blob, type);
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch {
      setError("Không truy cập được micro. Hãy cấp quyền micro cho trang này.");
    }
  };

  const stop = useCallback(() => {
    setRecording(false);
    if (recorderRef.current && recorderRef.current.state !== "inactive") recorderRef.current.stop();
  }, []);

  // Stop automatically at the server's maximum duration.
  const maxSeconds = status?.max_seconds || 120;
  useEffect(() => {
    if (recording && seconds >= maxSeconds) stop();
  }, [recording, seconds, maxSeconds, stop]);

  const unavailable = status && !status.available;
  const m = result?.metrics;

  return (
    <div className="page">
      <header className="page-head">
        <h1 className="page-title">Luyện nói</h1>
        <p className="page-sub">Nghe câu mẫu, ghi âm lại và nhận phản hồi về độ chính xác, tốc độ, ngắt nghỉ và độ rõ.</p>
      </header>
      {unavailable && (
        <p className="note note--err">
          Máy chủ chưa bật nhận dạng giọng nói. Cài <code>faster-whisper</code> và đặt <code>STT_ENABLED=true</code>.
        </p>
      )}
      {error && <p className="note note--err">{error}</p>}

      <section className="card card-pad" style={{ display: "grid", gap: 14 }}>
        <div className="exam-row">
          <div className="field">
            <label htmlFor="sp-lang">Ngôn ngữ</label>
            <select id="sp-lang" className="input" value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="en">English</option>
              <option value="vi">Tiếng Việt</option>
              <option value="auto">Tự nhận diện</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="sp-rate">Tốc độ đọc mẫu: {rate.toFixed(1)}×</label>
            <input id="sp-rate" type="range" min={0.6} max={1.2} step={0.1} value={rate} onChange={(e) => setRate(Number(e.target.value))} />
          </div>
        </div>
        <div className="field">
          <label htmlFor="sp-ref">Câu mẫu (tuỳ chọn — để trống nếu nói tự do)</label>
          <textarea id="sp-ref" className="input" rows={3} maxLength={2000} value={reference} onChange={(e) => setReference(e.target.value)}
            placeholder="The weather is lovely today, so I decided to go for a walk." />
        </div>
        <div className="actions">
          <button type="button" className="btn" onClick={speak} disabled={!reference.trim()}>Nghe mẫu</button>
          {!recording ? (
            <button type="button" className="btn btn--primary btn--lg" onClick={start} disabled={busy || unavailable}>
              {busy ? "Đang phân tích…" : "Bắt đầu ghi âm"}
            </button>
          ) : (
            <button type="button" className="btn btn--primary btn--lg rec" onClick={stop}>
              Dừng ({seconds}s / {maxSeconds}s)
            </button>
          )}
        </div>
        <p className="field-hint">Bản ghi chỉ được xử lý tạm thời để nhận dạng, không lưu trên máy chủ. Nghe mẫu dùng giọng đọc của trình duyệt.</p>
        {audioUrl && <audio controls src={audioUrl} style={{ width: "100%" }} />}
      </section>

      {result && (
        <section className="card card-pad" style={{ display: "grid", gap: 14, marginTop: 14 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <div className="result-score">{result.overall_score != null ? Math.round(result.overall_score) : "—"}</div>
            <span className="field-hint">Điểm tổng hợp /100</span>
          </div>
          <div>
            <h3 className="card-title">{result.reference ? "So với câu mẫu" : "Máy nghe được"}</h3>
            {result.reference ? <Alignment ops={result.reference.alignment} /> : <p>{result.transcript || "(không nghe rõ)"}</p>}
            {result.reference && (
              <p className="field-hint">
                Xanh: đúng · Đỏ: nói khác mẫu · Gạch/xám: bỏ sót · Tím: từ thừa. Độ chính xác {Math.round(result.reference.accuracy * 100)}% (WER {result.reference.wer}).
              </p>
            )}
          </div>
          <div className="metrics">
            <Metric label="Từ/phút" value={m.wpm} hint={`Khoảng thoải mái ${m.pace_band[0]}–${m.pace_band[1]}`} />
            <Metric label="Ngắt dài" value={m.long_pauses} hint="Khoảng lặng ≥ 0,7 giây" />
            <Metric label="Từ đệm" value={m.fillers} />
            <Metric label="Độ rõ" value={m.clarity != null ? `${Math.round(m.clarity)}%` : null}
              hint="Mức chắc chắn của máy nhận dạng, gợi ý cho độ rõ phát âm (không phải chấm âm vị)" />
          </div>
          {m.low_confidence_words.length > 0 && (
            <div>
              <h3 className="card-title">Từ nên luyện lại</h3>
              <div className="chips">
                {m.low_confidence_words.map((w, i) => <span key={i} className="badge badge--warn">{w.word} · {Math.round(w.confidence * 100)}%</span>)}
              </div>
            </div>
          )}
          {result.grammar?.suggestions?.length > 0 && (
            <div>
              <h3 className="card-title">Ngữ pháp</h3>
              <p className="explain">Gợi ý: {result.grammar.corrected}</p>
              <ul>{result.grammar.suggestions.map((s, i) => <li key={i}>{s}</li>)}</ul>
            </div>
          )}
          <div>
            <h3 className="card-title">Nhận xét</h3>
            <ul>{result.tips.map((t, i) => <li key={i}>{t}</li>)}</ul>
          </div>
        </section>
      )}
    </div>
  );
}
