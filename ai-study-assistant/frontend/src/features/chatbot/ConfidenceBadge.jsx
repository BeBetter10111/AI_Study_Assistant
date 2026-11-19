import { useState } from "react";

const LABEL = {
  high: { text: "Độ tin cậy cao", tone: "ok" },
  medium: { text: "Độ tin cậy trung bình", tone: "warn" },
  low: { text: "Độ tin cậy thấp", tone: "err" },
};

// Groundedness of the answer in the retrieved passages (see backend app/rag/verification.py). It is a heuristic
// estimate, not a proof of correctness, so low confidence means "double-check the cited sources".
export default function ConfidenceBadge({ confidence, label, unsupportedClaims = [], invalidCitations = [] }) {
  const [open, setOpen] = useState(false);
  const info = LABEL[label];
  if (!info || confidence == null) return null;
  const hasDetails = unsupportedClaims.length > 0 || invalidCitations.length > 0;
  return (
    <div className="confidence">
      <button
        type="button"
        className={`badge badge--${info.tone} confidence-btn`}
        onClick={() => hasDetails && setOpen((v) => !v)}
        aria-expanded={hasDetails ? open : undefined}
        title="Mức độ câu trả lời được các đoạn tài liệu hỗ trợ"
      >
        {info.text}: {Math.round(confidence * 100)}%{hasDetails ? (open ? " ▴" : " ▾") : ""}
      </button>
      {open && (
        <div className="confidence-detail">
          {unsupportedClaims.length > 0 && (
            <>
              <p className="field-hint">Các câu chưa tìm thấy căn cứ rõ ràng trong tài liệu:</p>
              <ul>
                {unsupportedClaims.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </>
          )}
          {invalidCitations.length > 0 && (
            <p className="field-hint">Trích dẫn không tồn tại: {invalidCitations.map((n) => `[${n}]`).join(", ")}</p>
          )}
        </div>
      )}
    </div>
  );
}
