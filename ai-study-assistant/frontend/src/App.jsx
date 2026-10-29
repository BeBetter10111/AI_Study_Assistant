import { useState, useEffect } from "react";
import {
  setAuthToken,
  uploadDocument,
  getDocumentStatus,
} from "./services/api";
import ChatWindow from "./features/chatbot/ChatWindow";
import QuizPlayer from "./features/quiz/QuizPlayer";
import FlashcardDeck from "./features/flashcard/FlashcardDeck";
import LoginPage from "./features/auth/LoginPage";
import SignupPage from "./features/auth/SignupPage";

const UPLOAD_CONCURRENCY = 3;
const ALLOWED_EXT = /\.(pdf|docx|pptx|md)$/i;
const DONE_STATUSES = ["embedded", "failed"];

const TABS = [
  { key: "chat", label: "Hỏi đáp" },
  { key: "quiz", label: "Quiz" },
  { key: "flashcard", label: "Thẻ ghi nhớ" },
];

// "login" | "signup"
const AUTH_SCREEN = { LOGIN: "login", SIGNUP: "signup" };

export default function App() {
  const [userEmail, setUserEmail] = useState(null);
  const [authScreen, setAuthScreen] = useState(AUTH_SCREEN.LOGIN);
  const [activeTab, setActiveTab] = useState("chat");
  const [documentId, setDocumentId] = useState(null); // null = search all documents
  const [uploads, setUploads] = useState([]); // [{key, name, id, status, error}]

  const handleAuthenticated = (email) => setUserEmail(email);

  const handleLogout = () => {
    setAuthToken(null);
    setUserEmail(null);
    setDocumentId(null);
    setUploads([]);
    setAuthScreen(AUTH_SCREEN.LOGIN);
  };

  const updateUpload = (key, patch) =>
    setUploads((prev) => prev.map((u) => (u.key === key ? { ...u, ...patch } : u)));

  const handleFilesChange = async (event) => {
    const files = Array.from(event.target.files || []).filter((f) =>
      ALLOWED_EXT.test(f.name)
    );
    event.target.value = ""; // allow choosing the same files again
    if (!files.length) return;

    const stamp = Date.now();
    const items = files.map((file, i) => ({
      key: `${stamp}-${i}-${file.name}`,
      file,
      name: file.name,
    }));
    setUploads((prev) => [
      ...prev,
      ...items.map(({ key, name }) => ({ key, name, id: null, status: "uploading", error: null })),
    ]);

    let next = 0;
    const worker = async () => {
      while (next < items.length) {
        const item = items[next++];
        try {
          const res = await uploadDocument(item.file);
          updateUpload(item.key, { id: res.data.document_id, status: res.data.status });
        } catch (err) {
          updateUpload(item.key, {
            status: "failed",
            error: err.response?.data?.detail || "Failed to upload file.",
          });
        }
      }
    };
    await Promise.all(
      Array.from({ length: Math.min(UPLOAD_CONCURRENCY, items.length) }, worker)
    );
  };

  useEffect(() => {
    const pending = uploads.filter((u) => u.id && !DONE_STATUSES.includes(u.status));
    if (!pending.length) return undefined;
    const timer = setInterval(() => {
      pending.forEach(async (u) => {
        try {
          const res = await getDocumentStatus(u.id);
          if (res.data.status !== u.status) updateUpload(u.key, { status: res.data.status });
        } catch {
          /* try again on the next tick */
        }
      });
    }, 3000);
    return () => clearInterval(timer);
  }, [uploads]);

  const embeddedCount = uploads.filter((u) => u.status === "embedded").length;

  if (!userEmail) {
    if (authScreen === AUTH_SCREEN.SIGNUP) {
      return (
        <SignupPage
          onAuthenticated={handleAuthenticated}
          onGoLogin={() => setAuthScreen(AUTH_SCREEN.LOGIN)}
        />
      );
    }
    return (
      <LoginPage
        onAuthenticated={handleAuthenticated}
        onGoSignup={() => setAuthScreen(AUTH_SCREEN.SIGNUP)}
      />
    );
  }

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>Explorer</h1>
        <div>
          <span style={{ marginRight: "12px" }}>{userEmail}</span>
          <button style={styles.secondaryButton} onClick={handleLogout}>
            Log out
          </button>
        </div>
      </header>

      <section style={styles.uploadRow}>
        <input type="file" multiple accept=".pdf,.docx,.pptx,.md" onChange={handleFilesChange} />
        <label style={styles.statusBadge}>
          Folder:{" "}
          <input type="file" multiple webkitdirectory="" onChange={handleFilesChange} />
        </label>
        {uploads.length > 0 && (
          <span style={styles.statusBadge}>
            {embeddedCount}/{uploads.length} ready
          </span>
        )}
        <select value={documentId || ""} onChange={(e) => setDocumentId(e.target.value || null)}>
          <option value="">All documents</option>
          {uploads
            .filter((u) => u.status === "embedded")
            .map((u) => (
              <option key={u.key} value={u.id}>
                {u.name}
              </option>
            ))}
        </select>
      </section>

      {uploads.length > 0 && (
        <ul style={styles.uploadList}>
          {uploads.map((u) => (
            <li key={u.key} style={u.status === "failed" ? styles.error : undefined}>
              {u.name} — {u.status}
              {u.error ? `: ${u.error}` : ""}
            </li>
          ))}
        </ul>
      )}

      <nav style={styles.tabs}>
        {TABS.map((tab) => (
          <button
            key={tab.key}
            style={{
              ...styles.tabButton,
              ...(activeTab === tab.key ? styles.tabButtonActive : {}),
            }}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main style={styles.main}>
        {activeTab === "chat" && <ChatWindow documentId={documentId} />}
        {activeTab === "quiz" && <QuizPlayer documentId={documentId} />}
        {activeTab === "flashcard" && <FlashcardDeck documentId={documentId} />}
      </main>
    </div>
  );
}

const styles = {
  page: {
    fontFamily: "system-ui, sans-serif",
    maxWidth: "760px",
    margin: "0 auto",
    padding: "16px",
    minHeight: "100vh",
    boxSizing: "border-box",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "12px",
  },
  title: { fontSize: "20px", margin: 0 },
  loginBox: {
    maxWidth: "320px",
    margin: "80px auto",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    padding: "24px",
    border: "1px solid #e2e8f0",
    borderRadius: "10px",
  },
  input: {
    padding: "8px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
  },
  primaryButton: {
    flex: 1,
    padding: "8px 16px",
    borderRadius: "6px",
    border: "none",
    backgroundColor: "#2563eb",
    color: "#fff",
    cursor: "pointer",
  },
  secondaryButton: {
    padding: "8px 16px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
    backgroundColor: "#fff",
    cursor: "pointer",
  },
  linkButton: {
    marginLeft: "8px",
    border: "none",
    background: "none",
    color: "#2563eb",
    cursor: "pointer",
    textDecoration: "underline",
  },
  error: { color: "#dc2626" },
  uploadRow: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    marginBottom: "12px",
    flexWrap: "wrap",
  },
  statusBadge: { fontSize: "14px", color: "#475569" },
  uploadList: { fontSize: "13px", color: "#475569", margin: "0 0 12px", paddingLeft: "18px" },
  tabs: { display: "flex", gap: "8px", marginBottom: "12px" },
  tabButton: {
    padding: "8px 16px",
    borderRadius: "6px",
    border: "1px solid #cbd5e1",
    backgroundColor: "#fff",
    cursor: "pointer",
  },
  tabButtonActive: { backgroundColor: "#2563eb", color: "#fff", borderColor: "#2563eb" },
  main: { minHeight: "420px" },
};
