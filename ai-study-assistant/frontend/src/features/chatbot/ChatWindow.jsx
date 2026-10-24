import { useState } from "react";
import { askQuestionStream } from "../../services/api";

export default function ChatWindow({ documentId = null }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [steps, setSteps] = useState([]);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isLoading) return;

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

    try {
      await askQuestionStream(question, documentId, (event) => {
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
          const { answer, citations, intent } = event.data;
          const finalMsg = { role: "assistant", content: answer, citations, intent, steps: collected };
          const wasStreaming = streamingStarted;
          setMessages((prev) => (wasStreaming ? [...prev.slice(0, -1), finalMsg] : [...prev, finalMsg]));
        } else if (event.type === "error") {
          failed = true;
          setError(event.detail);
        }
      });;
      if (!answered && !failed) {
        setError("No response received from the AI assistant.");
      }
    } catch (err) {
      setError(err.message || "An error occurred while connecting to the AI assistant.");
    } finally {
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

  return (
    <div style={styles.container}>
      <div style={styles.messageList}>
        {messages.length === 0 && (
          <p style={styles.placeholder}>
            Ask questions about the document you have uploaded to get started.
          </p>
        )}
        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              ...styles.messageBubble,
              ...(message.role === "user" ? styles.userBubble : styles.assistantBubble),
            }}
          >
            <p style={styles.messageText}>{message.content}</p>
            {message.steps && message.steps.length > 0 && (
              <details style={styles.stepsDetails}>
                <summary>Show steps</summary>
                <StepList steps={message.steps} />
              </details>
            )}
            {message.citations && message.citations.length > 0 && (
              <div style={styles.citations}>
                <span style={styles.citationsTitle}>Citations:</span>
                <ul style={styles.citationList}>
                  {message.citations.map((citation, cIndex) => (
                    <li key={cIndex} style={styles.citationItem}>
                      [{citation.chunk_index}] {citation.content.slice(0, 140)}
                      {citation.content.length > 140 ? "…" : ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div style={styles.liveSteps}>
            {steps.length === 0 ? <p style={styles.placeholder}>Starting…</p> : <StepList steps={steps} />}
          </div>
        )}
        {error && <p style={styles.error}>{error}</p>}
      </div>

      <div style={styles.inputRow}>
        <textarea
          style={styles.textarea}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter your question…"
          rows={2}
          disabled={isLoading}
        />
        <button
          style={styles.sendButton}
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}

function StepList({ steps }) {
  return (
    <ul style={styles.stepList}>
      {steps.map((step) => (
        <li key={step.id} style={styles.stepItem}>
          <span style={styles.stepIcon}>{step.status === "done" ? "✓" : "…"}</span>
          <span>
            {step.label}
            {step.status === "done" && step.ms != null && (
              <span style={styles.stepMs}> · {step.ms < 1000 ? `${step.ms} ms` : `${(step.ms / 1000).toFixed(1)} s`}</span>
            )}
            {step.detail && <div style={styles.stepDetail}>{step.detail}</div>}
          </span>
        </li>
      ))}
    </ul>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    overflow: "hidden",
  },
  messageList: {
    flex: 1,
    overflowY: "auto",
    padding: "12px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  placeholder: { color: "#94a3b8", fontStyle: "italic" },
  liveSteps: { alignSelf: "flex-start", fontSize: "13px", color: "#475569" },
  stepsDetails: { marginTop: "8px", fontSize: "12px", opacity: 0.85 },
  stepList: { listStyle: "none", margin: "4px 0 0", padding: 0 },
  stepItem: { display: "flex", gap: "6px", marginBottom: "4px" },
  stepIcon: { width: "14px", flexShrink: 0 },
  stepMs: { color: "#94a3b8" },
  stepDetail: { color: "#64748b", fontSize: "12px", wordBreak: "break-word" },
  error: { color: "#dc2626" },
  messageBubble: {
    maxWidth: "80%",
    padding: "10px 12px",
    borderRadius: "10px",
  },
  userBubble: {
    alignSelf: "flex-end",
    backgroundColor: "pink",
    color: "#fff",
  },
  assistantBubble: {
    alignSelf: "flex-start",
    backgroundColor: "#f1f5f9",
    color: "#0f172a",
  },
  messageText: { margin: 0, whiteSpace: "pre-wrap" },
  citations: { marginTop: "8px", fontSize: "12px", opacity: 0.85 },
  citationsTitle: { fontWeight: 600 },
  citationList: { margin: "4px 0 0 16px", padding: 0 },
  citationItem: { marginBottom: "4px" },
  inputRow: {
    display: "flex",
    gap: "8px",
    padding: "8px",
    borderTop: "1px solid #e2e8f0",
  },
  textarea: {
    flex: 1,
    resize: "none",
    padding: "8px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
    fontFamily: "inherit",
  },
  sendButton: {
    padding: "0 16px",
    borderRadius: "6px",
    border: "none",
    backgroundColor: "pink",
    color: "#fff",
    cursor: "pointer",
  },
};
