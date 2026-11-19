import { useCallback, useEffect, useState } from "react";
import {
  composePaper, deleteBankItem, deletePaper, downloadPaper, generateBankItems, getErrorMessage, gradeBankAnswer,
  listBankItems, listPapers, patchBankItem,
} from "../../services/api";

const QTYPE = { mcq: "Trắc nghiệm", short_answer: "Câu ngắn", essay: "Tự luận" };
const DIFF = { easy: { text: "Dễ", tone: "ok" }, medium: { text: "Trung bình", tone: "warn" }, hard: { text: "Khó", tone: "err" } };

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function BankItem({ item, selected, onToggle, onChanged, onError }) {
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [question, setQuestion] = useState(item.question);
  const [answerKey, setAnswerKey] = useState(item.answer_key);
  const [studentAnswer, setStudentAnswer] = useState("");
  const [grade, setGrade] = useState(null);
  const [busy, setBusy] = useState(false);
  const diff = DIFF[item.difficulty];
  const open_ = item.qtype !== "mcq";

  const run = async (fn, fallback) => {
    setBusy(true);
    try {
      await fn();
    } catch (err) {
      onError(getErrorMessage(err, fallback));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card card-pad bank-item">
      <div className="bank-head">
        <input type="checkbox" checked={selected} onChange={onToggle} aria-label="Chọn câu hỏi" />
        <button type="button" className="bank-title" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
          {item.question}
        </button>
      </div>
      <div className="bank-meta">
        <span className="badge">{QTYPE[item.qtype]}</span>
        <span className={`badge badge--${diff.tone}`}>{diff.text}</span>
        <span className="badge">{item.topic}</span>
        <span className={`badge badge--${item.status === "approved" ? "ok" : "warn"}`}>
          {item.status === "approved" ? "Đã duyệt" : "Nháp"}
        </span>
        {item.source_ref && <span className="field-hint">Nguồn: {item.source_ref}</span>}
      </div>
      {open && (
        <div style={{ display: "grid", gap: 10, marginTop: 10 }}>
          {editing ? (
            <>
              <textarea className="input" rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} />
              <textarea className="input" rows={3} value={answerKey} onChange={(e) => setAnswerKey(e.target.value)} placeholder="Đáp án / ý chính" />
              <div className="actions">
                <button type="button" className="btn btn--primary" disabled={busy || !question.trim()}
                  onClick={() => run(async () => {
                    await patchBankItem(item.id, { question, answer_key: answerKey });
                    setEditing(false);
                    onChanged();
                  }, "Không lưu được.")}>Lưu</button>
                <button type="button" className="btn" onClick={() => setEditing(false)}>Huỷ</button>
              </div>
            </>
          ) : (
            <>
              {item.options && (
                <div className="options">
                  {item.options.map((o, i) => (
                    <div key={i} className={`option${i === item.correct_index ? " option--correct" : ""}`}>
                      <span className="option-key">{String.fromCharCode(65 + i)}</span>
                      <span>{o}</span>
                    </div>
                  ))}
                </div>
              )}
              {open_ && <p className="explain"><b>Đáp án mẫu:</b> {item.answer_key}</p>}
              {item.explanation && <p className="explain">{item.explanation}</p>}
              {item.rubric && (
                <table className="mini-table">
                  <thead><tr><th>Tiêu chí</th><th>Điểm</th><th>Đạt tối đa</th><th>Một phần</th></tr></thead>
                  <tbody>
                    {item.rubric.map((c) => (
                      <tr key={c.criterion}><td>{c.criterion}</td><td>{c.max_points}</td><td>{c.full}</td><td>{c.partial}</td></tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
          <div className="actions">
            <button type="button" className="btn" disabled={busy}
              onClick={() => run(async () => {
                await patchBankItem(item.id, { status: item.status === "approved" ? "draft" : "approved" });
                onChanged();
              }, "Không cập nhật được.")}>
              {item.status === "approved" ? "Bỏ duyệt" : "Duyệt"}
            </button>
            {!editing && <button type="button" className="btn" onClick={() => setEditing(true)}>Sửa</button>}
            <button type="button" className="btn" disabled={busy}
              onClick={() => run(async () => { await deleteBankItem(item.id); onChanged(); }, "Không xoá được.")}>Xoá</button>
          </div>
          {open_ && item.rubric && (
            <div className="grade-box">
              <label className="field-hint" htmlFor={`ans-${item.id}`}>Chấm thử một bài làm theo rubric (AI chỉ gợi ý, giảng viên quyết định)</label>
              <textarea id={`ans-${item.id}`} className="input" rows={3} value={studentAnswer} onChange={(e) => setStudentAnswer(e.target.value)} />
              <div className="actions">
                <button type="button" className="btn" disabled={busy || !studentAnswer.trim()}
                  onClick={() => run(async () => setGrade((await gradeBankAnswer(item.id, studentAnswer)).data), "Không chấm được.")}>
                  {busy ? "Đang chấm…" : "Chấm thử"}
                </button>
              </div>
              {grade && (
                <div className="explain">
                  <b>Gợi ý: {grade.total}/{grade.max_total} điểm</b> (cần giảng viên xem lại)
                  <ul>
                    {grade.criteria.map((c) => (
                      <li key={c.criterion}>{c.criterion}: {c.points}/{c.max_points} — {c.justification}</li>
                    ))}
                  </ul>
                  {grade.feedback && <p>{grade.feedback}</p>}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Lecturer mode: generate a question bank from documents, assemble papers, export to Word.
export default function LecturerView({ documentId = null, scopeLabel = "Tất cả tài liệu" }) {
  const [tab, setTab] = useState("bank");
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [busy, setBusy] = useState(false);
  // bank
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [filters, setFilters] = useState({ qtype: "", difficulty: "", status: "", q: "" });
  const [selected, setSelected] = useState(new Set());
  // generate
  const [gen, setGen] = useState({ topic: "", qtype: "mcq", difficulty: "medium", count: 5, points: "" });
  // papers
  const [papers, setPapers] = useState([]);
  const [paper, setPaper] = useState({ title: "", mcq: 10, short_answer: 0, essay: 0, include_drafts: false, shuffle: true, mcq_points: 1 });

  const loadBank = useCallback(async () => {
    try {
      const params = Object.fromEntries(Object.entries(filters).filter(([, v]) => v));
      const res = await listBankItems({ ...params, limit: 100 });
      setItems(res.data.items);
      setTotal(res.data.total);
    } catch (err) {
      setError(getErrorMessage(err, "Không tải được ngân hàng câu hỏi."));
    }
  }, [filters]);

  const loadPapers = useCallback(async () => {
    try {
      setPapers((await listPapers()).data);
    } catch (err) {
      setError(getErrorMessage(err, "Không tải được danh sách đề."));
    }
  }, []);

  useEffect(() => { if (tab === "bank") loadBank(); }, [tab, loadBank]);
  useEffect(() => { if (tab === "papers") loadPapers(); }, [tab, loadPapers]);

  const wrap = async (fn, fallback) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (err) {
      setError(getErrorMessage(err, fallback));
    } finally {
      setBusy(false);
    }
  };

  const generate = () => wrap(async () => {
    const res = await generateBankItems({
      topic: gen.topic, qtype: gen.qtype, difficulty: gen.difficulty, count: Number(gen.count) || 5,
      points: gen.points ? Number(gen.points) : null, document_id: documentId,
    });
    setNotice(`Đã tạo ${res.data.items.length} câu (trạng thái Nháp). Hãy xem lại và duyệt trước khi dùng.`);
    setTab("bank");
  }, "Không tạo được câu hỏi.");

  const composeFromSelection = () => {
    const title = window.prompt("Tên đề thi:", "Đề thi mới");
    if (!title) return;
    wrap(async () => {
      await composePaper({ title, item_ids: [...selected], shuffle: true });
      setSelected(new Set());
      setNotice("Đã tạo đề. Xem ở tab Đề thi.");
    }, "Không tạo được đề.");
  };

  const composeAuto = () => wrap(async () => {
    await composePaper({
      title: paper.title || "Đề thi mới",
      auto: { mcq: Number(paper.mcq) || 0, short_answer: Number(paper.short_answer) || 0, essay: Number(paper.essay) || 0, include_drafts: paper.include_drafts },
      shuffle: paper.shuffle, mcq_points: Number(paper.mcq_points) || 1,
    });
    setNotice("Đã tạo đề tự động.");
    loadPapers();
  }, "Không tạo được đề tự động.");

  const download = (p, format, answers) => wrap(async () => {
    const res = await downloadPaper(p.id, format, answers);
    saveBlob(res.data, `${p.title}${answers ? "_dapan" : ""}.${format}`);
  }, "Không tải được file.");

  return (
    <div className="page">
      <header className="page-head">
        <h1 className="page-title">Chế độ giảng viên</h1>
        <p className="page-sub">Sinh câu hỏi và rubric từ tài liệu, duyệt vào ngân hàng, ghép đề và xuất Word. Dữ liệu chỉ của riêng bạn.</p>
      </header>
      <div className="tabs" role="tablist">
        {[["bank", "Ngân hàng câu hỏi"], ["generate", "Tạo câu hỏi"], ["papers", "Đề thi"]].map(([key, label]) => (
          <button key={key} type="button" role="tab" aria-selected={tab === key} className="tab" onClick={() => setTab(key)}>{label}</button>
        ))}
      </div>
      {error && <p className="note note--err">{error}</p>}
      {notice && <p className="note">{notice}</p>}

      {tab === "generate" && (
        <section className="card card-pad" style={{ display: "grid", gap: 14 }}>
          <div className="field">
            <label htmlFor="lec-topic">Chủ đề</label>
            <input id="lec-topic" className="input" value={gen.topic} maxLength={200} onChange={(e) => setGen({ ...gen, topic: e.target.value })} />
          </div>
          <div className="exam-row">
            <div className="field">
              <label htmlFor="lec-type">Loại câu</label>
              <select id="lec-type" className="input" value={gen.qtype} onChange={(e) => setGen({ ...gen, qtype: e.target.value })}>
                {Object.entries(QTYPE).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </div>
            <div className="field">
              <label htmlFor="lec-diff">Độ khó</label>
              <select id="lec-diff" className="input" value={gen.difficulty} onChange={(e) => setGen({ ...gen, difficulty: e.target.value })}>
                {Object.entries(DIFF).map(([k, v]) => <option key={k} value={k}>{v.text}</option>)}
              </select>
            </div>
            <div className="field">
              <label htmlFor="lec-count">Số câu</label>
              <input id="lec-count" type="number" min={1} max={20} className="input input--num" value={gen.count} onChange={(e) => setGen({ ...gen, count: e.target.value })} />
            </div>
            {gen.qtype !== "mcq" && (
              <div className="field">
                <label htmlFor="lec-pts">Tổng điểm/câu</label>
                <input id="lec-pts" type="number" min={0.5} step={0.5} className="input input--num" value={gen.points}
                  placeholder={gen.qtype === "essay" ? "10" : "4"} onChange={(e) => setGen({ ...gen, points: e.target.value })} />
              </div>
            )}
          </div>
          <p className="field-hint">Phạm vi tài liệu: {scopeLabel}. Câu hỏi tạo ra ở trạng thái Nháp để bạn kiểm tra trước khi dùng.</p>
          <div className="actions">
            <button type="button" className="btn btn--primary btn--lg" disabled={busy || !gen.topic.trim()} onClick={generate}>
              {busy ? "Đang tạo…" : "Tạo câu hỏi"}
            </button>
          </div>
        </section>
      )}

      {tab === "bank" && (
        <>
          <div className="card card-pad filters">
            <select className="input" aria-label="Loại câu" value={filters.qtype} onChange={(e) => setFilters({ ...filters, qtype: e.target.value })}>
              <option value="">Mọi loại</option>
              {Object.entries(QTYPE).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <select className="input" aria-label="Độ khó" value={filters.difficulty} onChange={(e) => setFilters({ ...filters, difficulty: e.target.value })}>
              <option value="">Mọi độ khó</option>
              {Object.entries(DIFF).map(([k, v]) => <option key={k} value={k}>{v.text}</option>)}
            </select>
            <select className="input" aria-label="Trạng thái" value={filters.status} onChange={(e) => setFilters({ ...filters, status: e.target.value })}>
              <option value="">Mọi trạng thái</option>
              <option value="draft">Nháp</option>
              <option value="approved">Đã duyệt</option>
            </select>
            <input className="input" aria-label="Tìm câu hỏi" placeholder="Tìm…" value={filters.q} onChange={(e) => setFilters({ ...filters, q: e.target.value })} />
          </div>
          <p className="field-hint" style={{ margin: "8px 2px" }}>{total} câu · đã chọn {selected.size}</p>
          {items.map((it) => (
            <BankItem
              key={it.id}
              item={it}
              selected={selected.has(it.id)}
              onToggle={() => setSelected((s) => { const n = new Set(s); if (n.has(it.id)) n.delete(it.id); else n.add(it.id); return n; })}
              onChanged={loadBank}
              onError={setError}
            />
          ))}
          {items.length === 0 && <p className="field-hint">Chưa có câu hỏi. Dùng tab “Tạo câu hỏi”.</p>}
          {selected.size > 0 && (
            <div className="actions" style={{ marginTop: 12 }}>
              <button type="button" className="btn btn--primary btn--lg" disabled={busy} onClick={composeFromSelection}>
                Tạo đề từ {selected.size} câu đã chọn
              </button>
            </div>
          )}
        </>
      )}

      {tab === "papers" && (
        <>
          <section className="card card-pad" style={{ display: "grid", gap: 12 }}>
            <h3 className="card-title">Ghép đề tự động (từ câu đã duyệt)</h3>
            <div className="field">
              <label htmlFor="paper-title">Tên đề</label>
              <input id="paper-title" className="input" value={paper.title} onChange={(e) => setPaper({ ...paper, title: e.target.value })} />
            </div>
            <div className="exam-row">
              {[["mcq", "Trắc nghiệm"], ["short_answer", "Câu ngắn"], ["essay", "Tự luận"]].map(([k, label]) => (
                <div className="field" key={k}>
                  <label htmlFor={`p-${k}`}>{label}</label>
                  <input id={`p-${k}`} type="number" min={0} className="input input--num" value={paper[k]} onChange={(e) => setPaper({ ...paper, [k]: e.target.value })} />
                </div>
              ))}
              <div className="field">
                <label htmlFor="p-pts">Điểm/câu TN</label>
                <input id="p-pts" type="number" min={0.25} step={0.25} className="input input--num" value={paper.mcq_points} onChange={(e) => setPaper({ ...paper, mcq_points: e.target.value })} />
              </div>
            </div>
            <label className="check"><input type="checkbox" checked={paper.shuffle} onChange={(e) => setPaper({ ...paper, shuffle: e.target.checked })} /> Xáo trộn đáp án</label>
            <label className="check"><input type="checkbox" checked={paper.include_drafts} onChange={(e) => setPaper({ ...paper, include_drafts: e.target.checked })} /> Cho phép cả câu Nháp</label>
            <div className="actions">
              <button type="button" className="btn btn--primary" disabled={busy} onClick={composeAuto}>Tạo đề</button>
            </div>
          </section>
          <section className="card card-pad">
            {papers.length === 0 ? <p className="field-hint">Chưa có đề nào.</p> : papers.map((p) => (
              <div key={p.id} className="list-row">
                <div>
                  <b>{p.title}</b>
                  <p className="field-hint">{p.num_questions} câu · {p.total_points} điểm</p>
                </div>
                <div className="actions">
                  <button type="button" className="btn" disabled={busy} onClick={() => download(p, "docx", false)}>Word</button>
                  <button type="button" className="btn" disabled={busy} onClick={() => download(p, "docx", true)}>Word + đáp án</button>
                  <button type="button" className="btn" disabled={busy} onClick={() => download(p, "md", true)}>Markdown</button>
                  <button type="button" className="btn" onClick={() => wrap(async () => { await deletePaper(p.id); loadPapers(); }, "Không xoá được.")}>Xoá</button>
                </div>
              </div>
            ))}
          </section>
        </>
      )}
    </div>
  );
}
