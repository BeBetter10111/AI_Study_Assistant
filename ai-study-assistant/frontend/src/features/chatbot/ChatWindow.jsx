import { useEffect, useRef, useState } from "react";
import Icon from "../../components/Icons";
import { askQuestionStream } from "../../services/api";

export default function ChatWindow({ documentId = null, scopeLabel = "Tất cả tài liệu", hasDocuments = false }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [steps, setSteps] = useState([]);
  const textareaRef = useRef(null);
  const scrollRef = useRef(null);
  const wasLoading = useRef(false);
  const abortRef = useRef(null);

  // Ô nhập tự cao lên theo nội dung (tối đa ~6 dòng).
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [input]);

  // Luôn cuộn tới tin mới nhất; trả lại focus cho ô nhập khi trợ lý trả lời xong.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, steps, error]);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    if (wasLoading.current && !isLoading) textareaRef.current?.focus();
    wasLoading.current = isLoading;
  }, [isLoading]);

  // Lịch sử gửi kèm để server viết lại câu hỏi nối tiếp ("còn cái đó thì sao?") thành câu hỏi độc lập.
  // Chỉ lấy các lượt đã hoàn chỉnh, tối đa 6 tin gần nhất.
  const buildHistory = () =>
    messages
      .filter((m) => m.content && !m.partial)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));

  const handleStop = () => abortRef.current?.abort();

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading) return;

    const history = buildHistory();
    const controller = new AbortController();
    abortRef.current = controller;

    const userMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);
    setSteps([]);

    let collected = [];
    let answered = false;
    let failed = false;
    let streamingStarted = false;
    const upsertStep = (step) => {
      const i = collected.findIndex((s) => s.id === step.id);
      collected = i === -1 ? [...collected, step] : collected.map((s, j) => (j === i ? step : s));
      setSteps(collected);
    };
    // Câu trả lời đang stream dở mà bị dừng / lỗi: giữ lại phần đã nhận và đánh dấu "chưa hoàn chỉnh".
    const markPartial = () => {
      if (!streamingStarted) return;
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        return last?.role === "assistant" && !last.citations
          ? [...prev.slice(0, -1), { ...last, partial: true }]
          : prev;
      });
    };

    try {
      await askQuestionStream(
        question,
        documentId,
        (event) => {
          if (event.type === "step") {
            upsertStep(event);
          } else if (event.type === "token") {
            if (!streamingStarted) {
              streamingStarted = true;
              setMessages((prev) => [...prev, { role: "assistant", content: event.text }]);
            } else {
              setMessages((prev) => {
                const last = prev[prev.length - 1];
                return [...prev.slice(0, -1), { ...last, content: last.content + event.text }];
              });
            }
          } else if (event.type === "result") {
            answered = true;
            const { answer, citations, intent, cached } = event.data;
            const finalMsg = { role: "assistant", content: answer, citations, intent, cached, steps: collected };
            const wasStreaming = streamingStarted;
            setMessages((prev) => (wasStreaming ? [...prev.slice(0, -1), finalMsg] : [...prev, finalMsg]));
          } else if (event.type === "error") {
            failed = true;
            markPartial();
            setError(event.detail);
          }
        },
        { history, signal: controller.signal }
      );
      if (!answered && !failed) {
        markPartial();
        setError("Trợ lý không trả về câu trả lời. Vui lòng thử lại.");
      }
    } catch (err) {
      markPartial();
      // Người dùng bấm "Dừng": không phải lỗi, chỉ giữ phần đã nhận.
      if (err?.name !== "AbortError") {
        setError(err.message || "An error occurred while connecting to the AI assistant.");
      }
    } finally {
      abortRef.current = null;
      setIsLoading(false);
      setSteps([]);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const isEmpty = messages.length === 0 && !isLoading && !error;

  return (
    <div className={`chat${isEmpty ? " chat--empty" : ""}`}>
      <div className="chat-scroll" ref={scrollRef}>
        {isEmpty ? (
          <div className="hero">
            <h1 className="hero-title">Hôm nay bạn muốn ôn gì?</h1>
            <p className="hero-sub">
              {hasDocuments
                ? "Hỏi bất kỳ điều gì về tài liệu đã tải lên."
                : "Tải tài liệu lên ở thanh bên rồi đặt câu hỏi về nội dung."}
            </p>
          </div>
        ) : (
          <div className="chat-col">
            {messages.map((message, index) =>
              message.role === "user" ? (
                <div key={index} className="msg-user">
                  {message.content}
                </div>
              ) : (
                <div key={index}>
                  <div className="msg-ai">{message.content}</div>
                  {message.partial && <p className="msg-extra">Câu trả lời chưa hoàn chỉnh (đã dừng hoặc bị gián đoạn).</p>}
                  {message.cached && (
                    <p className="msg-extra">Trả lời từ một câu hỏi tương tự đã hỏi trước đó.</p>
                  )}
                  {message.steps && message.steps.length > 0 && (
                    <details className="msg-extra">
                      <summary>Xem các bước xử lý</summary>
                      <StepList steps={message.steps} />
                    </details>
                  )}
                  {message.citations && message.citations.length > 0 && (
                    <details className="msg-extra">
                      <summary>Nguồn trích dẫn ({message.citations.length})</summary>
                      <ul className="sources">
                        {message.citations.map((citation, cIndex) => (
                          <li key={cIndex} className="source">
                            <b>[{citation.chunk_index}]</b>
                            {citation.content.slice(0, 140)}
                            {citation.content.length > 140 ? "…" : ""}
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              )
            )}
            {isLoading && (
              <div className="live msg-extra" role="status">
                {steps.length === 0 ? (
                  <span className="step">
                    <span className="spinner" /> Đang bắt đầu…
                  </span>
                ) : (
                  <StepList steps={steps} />
                )}
              </div>
            )}
            {error && (
              <p className="note note--err chat-error" role="alert">
                {error}
              </p>
            )}
          </div>
        )}
      </div>

      <div className="composer-wrap">
        <div className="composer">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Đặt câu hỏi về tài liệu của bạn…"
            aria-label="Câu hỏi"
            rows={1}
            disabled={isLoading}
          />
          <div className="composer-bar">
            <span className="scope" title={scopeLabel}>
              {scopeLabel}
            </span>
            {isLoading ? (
              <button type="button" className="send" onClick={handleStop} aria-label="Dừng trả lời" title="Dừng">
                <Icon name="stop" size={17} />
              </button>
            ) : (
              <button
                type="button"
                className="send"
                onClick={handleSend}
                disabled={!input.trim()}
                aria-label="Gửi câu hỏi"
              >
                <Icon name="send" size={17} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StepList({ steps }) {
  return (
    <ul className="steps">
      {steps.map((step) => (
        <li key={step.id} className="step">
          {step.status === "done" ? <Icon name="check" size={14} /> : <span className="spinner" />}
          <span>
            {step.label}
            {step.status === "done" && step.ms != null && (
              <span> · {step.ms < 1000 ? `${step.ms} ms` : `${(step.ms / 1000).toFixed(1)} s`}</span>
            )}
            {step.detail && <span className="step-detail">{step.detail}</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}
