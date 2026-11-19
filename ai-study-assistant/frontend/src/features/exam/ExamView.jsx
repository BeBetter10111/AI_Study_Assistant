import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteExam, generateExam, getErrorMessage, getExam, getWeakTopics, listExams, submitExam,
} from "../../services/api";

const PHASE = { HOME: "home", TAKING: "taking", RESULT: "result" };
const pct = (v) => `${Math.round(v * 100)}%`;
const fmtTime = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

function Bar({ value, low, high, weak }) {
  return (
    <div className="bar" aria-hidden="true">
      {low != null && <div className="bar-range" style={{ left: pct(low), width: pct(Math.max(high - low, 0.01)) }} />}
      <div className={`bar-fill${weak ? " bar-fill--weak" : ""}`} style={{ width: pct(value) }} />
    </div>
  );
}

// Exam-prep mode: syllabus -> exam -> server-side grading -> per-topic report.
export default function ExamView({ documentId = null, scopeLabel = "Tất cả tài liệu", onPracticeTopic }) {
  const [tab, setTab] = useState("new");
  const [phase, setPhase] = useState(PHASE.HOME);
  const [title, setTitle] = useState("");
  const [syllabus, setSyllabus] = useState("");
  const [numQuestions, setNumQuestions] = useState(10);
  const [duration, setDuration] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [exam, setExam] = useState(null);
  const [answers, setAnswers] = useState({});
  const [secondsLeft, setSecondsLeft] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [report, setReport] = useState(null);
  const startedAt = useRef(0);
  const submittingRef = useRef(false);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory((await listExams()).data);
    } catch (err) {
      setError(getErrorMessage(err, "Không tải được danh sách đề thi."));
    }
  }, []);

  useEffect(() => {
    if (tab === "history") refreshHistory();
    if (tab === "report") {
      getWeakTopics()
        .then((res) => setReport(res.data))
        .catch((err) => setError(getErrorMessage(err, "Không tải được báo cáo.")));
    }
  }, [tab, refreshHistory]);

  const begin = (e) => {
    setExam(e);
    setAnswers({});
    setResult(null);
    setError(null);
    setSecondsLeft(e.duration_min ? e.duration_min * 60 : null);
    startedAt.current = Date.now();
    submittingRef.current = false;
    setPhase(PHASE.TAKING);
  };

  const handleGenerate = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await generateExam({
        title,
        syllabus,
        num_questions: Number(numQuestions) || 10,
        duration_min: duration ? Number(duration) : null,
        document_id: documentId,
      });
      begin(res.data);
    } catch (err) {
      setError(getErrorMessage(err, "Không tạo được đề thi."));
    } finally {
      setBusy(false);
    }
  };

  const handleSubmit = useCallback(async () => {
    if (!exam || submittingRef.current) return;
    submittingRef.current = true;
    setBusy(true);
    try {
      const payload = Object.entries(answers).map(([index, choice]) => ({ index: Number(index), choice }));
      const res = await submitExam(exam.id, payload, Math.round((Date.now() - startedAt.current) / 1000));
      setResult(res.data);
      setPhase(PHASE.RESULT);
    } catch (err) {
      submittingRef.current = false;
      setError(getErrorMessage(err, "Không nộp được bài."));
    } finally {
      setBusy(false);
    }
  }, [exam, answers]);

  // Countdown; auto-submit when time is up.
  useEffect(() => {
    if (phase !== PHASE.TAKING || secondsLeft == null) return undefined;
    if (secondsLeft <= 0) {
      handleSubmit();
      return undefined;
    }
    const t = setTimeout(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [phase, secondsLeft, handleSubmit]);

  const retake = async (id) => {
    try {
      begin((await getExam(id)).data);
      setTab("new");
    } catch (err) {
      setError(getErrorMessage(err, "Không mở được đề thi."));
    }
  };

  const remove = async (id) => {
    try {
      await deleteExam(id);
      refreshHistory();
    } catch (err) {
      setError(getErrorMessage(err, "Không xoá được đề thi."));
    }
  };

  // ------------------------------------------------------------------ TAKING
  if (phase === PHASE.TAKING && exam) {
    const answered = Object.keys(answers).length;
    return (
      <div className="page">
        <header className="page-head">
          <h1 className="page-title">{exam.title}</h1>
          <p className="page-sub">
            {answered}/{exam.num_questions} câu đã trả lời
            {secondsLeft != null && (
              <span className={`badge ${secondsLeft < 60 ? "badge--err" : "badge--ok"}`} style={{ marginLeft: 10 }}>
                Còn {fmtTime(secondsLeft)}
              </span>
            )}
          </p>
        </header>
        {error && <p className="note note--err">{error}</p>}
        {exam.skipped_topics?.length > 0 && (
          <p className="note">Chủ đề chưa có nội dung trong tài liệu nên bị bỏ qua: {exam.skipped_topics.join(", ")}.</p>
        )}
        {exam.questions.map((q) => (
          <section key={q.index} className="card card-pad exam-q">
            <p className="field-hint">
              Câu {q.index + 1} · {q.topic}
            </p>
            <h2 className="question">{q.question}</h2>
            <div className="options">
              {q.options.map((opt, i) => (
                <button
                  key={i}
                  type="button"
                  className={`option${answers[q.index] === i ? " option--selected" : ""}`}
                  aria-pressed={answers[q.index] === i}
                  onClick={() => setAnswers((a) => ({ ...a, [q.index]: i }))}
                >
                  <span className="option-key">{String.fromCharCode(65 + i)}</span>
                  <span>{opt}</span>
                </button>
              ))}
            </div>
          </section>
        ))}
        <div className="actions">
          <button type="button" className="btn btn--primary btn--lg" disabled={busy} onClick={handleSubmit}>
            {busy ? "Đang chấm…" : "Nộp bài"}
          </button>
        </div>
      </div>
    );
  }

  // ------------------------------------------------------------------ RESULT
  if (phase === PHASE.RESULT && result) {
    return (
      <div className="page">
        <header className="page-head">
          <h1 className="page-title">Kết quả</h1>
          <p className="page-sub">Chấm điểm trên máy chủ, đáp án chỉ hiện sau khi nộp.</p>
        </header>
        <section className="card card-pad" style={{ display: "grid", gap: 14 }}>
          <div className="result-score">{result.score}%</div>
          <p className="page-sub">
            Đúng {result.correct}/{result.total} câu
            {result.unanswered > 0 ? ` · bỏ trống ${result.unanswered}` : ""}
            {result.overtime ? " · quá giờ" : ""}
          </p>
          <h3 className="card-title">Theo chủ đề</h3>
          {result.per_topic.map((t) => (
            <div key={t.topic} className="topic-row">
              <span>{t.topic}</span>
              <Bar value={t.accuracy} weak={t.weak} />
              <span className={`badge badge--${t.weak ? "err" : "ok"}`}>
                {t.correct}/{t.total}
              </span>
            </div>
          ))}
          {result.weak_topics.length > 0 && (
            <div>
              <h3 className="card-title">Cần ôn lại</h3>
              <div className="actions">
                {result.weak_topics.map((t) => (
                  <button key={t} type="button" className="btn" onClick={() => onPracticeTopic?.(t)}>
                    Luyện quiz: {t}
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
        <h3 className="card-title" style={{ margin: "18px 0 8px" }}>Xem lại đáp án</h3>
        {result.review.map((r) => (
          <section key={r.index} className="card card-pad exam-q">
            <p className="field-hint">
              Câu {r.index + 1} · {r.topic} ·{" "}
              <span className={`badge badge--${r.is_correct ? "ok" : "err"}`}>{r.is_correct ? "Đúng" : "Sai"}</span>
            </p>
            <h2 className="question">{r.question}</h2>
            <div className="options">
              {r.options.map((opt, i) => {
                let tone = "";
                if (i === r.correct_index) tone = " option--correct";
                else if (i === r.your_choice) tone = " option--wrong";
                return (
                  <div key={i} className={`option${tone}`}>
                    <span className="option-key">{String.fromCharCode(65 + i)}</span>
                    <span>{opt}</span>
                  </div>
                );
              })}
            </div>
            {r.explanation && <p className="explain">{r.explanation}</p>}
          </section>
        ))}
        <div className="actions">
          <button type="button" className="btn btn--primary btn--lg" onClick={() => setPhase(PHASE.HOME)}>
            Xong
          </button>
          <button type="button" className="btn btn--lg" onClick={() => begin(exam)}>
            Làm lại đề này
          </button>
        </div>
      </div>
    );
  }

  // ------------------------------------------------------------------ HOME
  return (
    <div className="page">
      <header className="page-head">
        <h1 className="page-title">Ôn thi</h1>
        <p className="page-sub">Tạo đề từ đề cương, làm bài có giới hạn thời gian và xem báo cáo chủ đề yếu.</p>
      </header>
      <div className="tabs" role="tablist">
        {[["new", "Tạo đề"], ["history", "Đề đã tạo"], ["report", "Chủ đề yếu"]].map(([key, label]) => (
          <button key={key} type="button" role="tab" aria-selected={tab === key} className="tab" onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </div>
      {error && <p className="note note--err">{error}</p>}

      {tab === "new" && (
        <section className="card card-pad" style={{ display: "grid", gap: 14 }}>
          <div className="field">
            <label htmlFor="exam-title">Tên đề (tuỳ chọn)</label>
            <input id="exam-title" className="input" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={200} />
          </div>
          <div className="field">
            <label htmlFor="exam-syllabus">Đề cương (mỗi chủ đề một dòng)</label>
            <textarea
              id="exam-syllabus"
              className="input"
              rows={6}
              value={syllabus}
              onChange={(e) => setSyllabus(e.target.value)}
              placeholder={"1. Stack và Queue\n2. Danh sách liên kết\n3. Cây nhị phân"}
            />
          </div>
          <div className="exam-row">
            <div className="field">
              <label htmlFor="exam-n">Số câu</label>
              <input id="exam-n" type="number" min={1} max={40} className="input input--num" value={numQuestions}
                onChange={(e) => setNumQuestions(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="exam-d">Thời gian (phút, tuỳ chọn)</label>
              <input id="exam-d" type="number" min={1} max={600} className="input input--num" value={duration}
                onChange={(e) => setDuration(e.target.value)} />
            </div>
          </div>
          <p className="field-hint">Phạm vi tài liệu: {scopeLabel}. Câu hỏi chỉ được tạo từ nội dung có trong tài liệu.</p>
          <div className="actions">
            <button type="button" className="btn btn--primary btn--lg" disabled={busy || syllabus.trim().length < 3} onClick={handleGenerate}>
              {busy ? "Đang tạo đề…" : "Tạo đề thi"}
            </button>
          </div>
        </section>
      )}

      {tab === "history" && (
        <section className="card card-pad">
          {history.length === 0 ? (
            <p className="field-hint">Chưa có đề nào.</p>
          ) : (
            history.map((h) => (
              <div key={h.id} className="list-row">
                <div>
                  <b>{h.title}</b>
                  <p className="field-hint">
                    {h.num_questions} câu · {h.attempts} lần làm
                    {h.best_score != null ? ` · cao nhất ${h.best_score}%` : ""}
                    {h.last_score != null ? ` · gần nhất ${h.last_score}%` : ""}
                  </p>
                </div>
                <div className="actions">
                  <button type="button" className="btn" onClick={() => retake(h.id)}>Làm bài</button>
                  <button type="button" className="btn" onClick={() => remove(h.id)}>Xoá</button>
                </div>
              </div>
            ))
          )}
        </section>
      )}

      {tab === "report" && (
        <section className="card card-pad" style={{ display: "grid", gap: 12 }}>
          {!report ? (
            <p className="field-hint">Đang tải…</p>
          ) : report.topics.length === 0 ? (
            <p className="field-hint">Chưa có dữ liệu. Hãy làm ít nhất một đề thi.</p>
          ) : (
            <>
              <p className="field-hint">
                Dựa trên {report.attempts_considered} lần làm bài. Độ chính xác được làm mượt (Laplace) và hiển thị khoảng tin cậy Wilson;
                chủ đề có dưới 3 câu chỉ là tham khảo.
              </p>
              {report.topics.map((t) => (
                <div key={t.topic} className="topic-row">
                  <span>{t.topic}</span>
                  <Bar value={t.smoothed_accuracy} low={t.wilson_low} high={t.wilson_high} weak={t.weak} />
                  <span className={`badge badge--${t.weak ? "err" : t.needs_more_data ? "warn" : "ok"}`}>
                    {t.weak ? "Yếu" : t.needs_more_data ? "Cần thêm dữ liệu" : "Ổn"} · {t.correct}/{t.total}
                  </span>
                </div>
              ))}
              {report.recommendations.length > 0 && (
                <div className="actions">
                  {report.recommendations.map((r) => (
                    <button key={r.topic} type="button" className="btn" onClick={() => onPracticeTopic?.(r.topic)}>
                      Luyện quiz: {r.topic}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </section>
      )}
    </div>
  );
}
