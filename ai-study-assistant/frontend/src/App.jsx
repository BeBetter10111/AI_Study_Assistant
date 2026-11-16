import { useCallback, useEffect, useState } from "react";
import {
  getErrorMessage,
  getMe,
  setAuthToken,
  setUnauthorizedHandler,
  uploadDocument,
  getDocumentStatus,
  listDocuments,
  deleteDocument,
  listConversations,
  getConversation,
  deleteConversation,
} from "./services/api";
import ChatWindow from "./features/chatbot/ChatWindow";
import QuizPlayer from "./features/quiz/QuizPlayer";
import FlashcardDeck from "./features/flashcard/FlashcardDeck";
import LoginPage from "./features/auth/LoginPage";
import SignupPage from "./features/auth/SignupPage";
import ForgotPasswordPage from "./features/auth/ForgotPasswordPage";
import VerifyCodePage from "./features/auth/VerifyCodePage";
import ResetPasswordPage from "./features/auth/ResetPasswordPage";
import ProfilePage from "./features/profile/ProfilePage";
import SettingsPage from "./features/settings/SettingsPage";
import Sidebar from "./components/Sidebar";
import ConfirmDialog from "./components/ConfirmDialog";
import Icon from "./components/Icons";

const UPLOAD_CONCURRENCY = 3;
const ALLOWED_EXT = /\.(pdf|docx|pptx|md)$/i;
const DONE_STATUSES = ["embedded", "failed"];

// Screens available after login. Store the route in the URL hash (#/profile) so reloads stay on the same screen.
const VIEWS = ["chat", "quiz", "flashcard", "profile", "settings"];
const readView = () => {
  const name = window.location.hash.replace(/^#\/?/, "");
  return VIEWS.includes(name) ? name : "chat";
};

// The server list is the source of truth; retain local-only items (uploading / failed uploads).
const mergeServerDocs = (prev, docs) => {
  const known = new Set(docs.map((d) => d.id));
  const local = prev.filter((u) => !u.id || !known.has(u.id));
  const server = docs.map((d) => ({ key: d.id, id: d.id, name: d.filename, status: d.status, error: null }));
  return [...local, ...server];
};

// Convert server-stored messages to the format used by ChatWindow.
const toUiMessage = (m) =>
  m.role === "assistant"
    ? { role: "assistant", content: m.content, citations: m.citations || [], intent: m.intent, cached: m.cached }
    : { role: "user", content: m.content };

const AUTH_SCREEN = {
  LOGIN: "login",
  SIGNUP: "signup",
  FORGOT: "forgot",
  VERIFY: "verify",
  RESET: "reset",
};

function Navigate({ to, onNavigate }) {
  useEffect(() => {
    onNavigate(to);
  }, [to, onNavigate]);
  return null;
}

export default function App() {
  const [userEmail, setUserEmail] = useState(() => {
    const savedToken = localStorage.getItem("auth_token");
    const savedEmail = localStorage.getItem("auth_email");
    if (savedToken && savedEmail) {
      setAuthToken(savedToken);
      return savedEmail;
    }
    return null;
  });
  const [authScreen, setAuthScreen] = useState(AUTH_SCREEN.LOGIN);
  const [resetEmail, setResetEmail] = useState("");
  const [resetToken, setResetToken] = useState(null); 
  const [loginNotice, setLoginNotice] = useState(null);
  const [view, setView] = useState(readView);
  const [navOpen, setNavOpen] = useState(false);
  const [confirmingLogout, setConfirmingLogout] = useState(false);
  const [pendingDelete, setPendingDelete] = useState(null); // {type: "document" | "conversation", id, name}
  const [notice, setNotice] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null); // null = new conversation
  const [initialMessages, setInitialMessages] = useState([]);
  const [chatKey, setChatKey] = useState(0); // Changing this remounts ChatWindow with a different conversation.
  const [profile, setProfile] = useState(null);
  const [documentId, setDocumentId] = useState(null);
  const [uploads, setUploads] = useState([]); 

  useEffect(() => {
    const onHash = () => setView(readView());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = (next) => {
    setNavOpen(false);
    window.location.hash = `/${next}`;
  };

  // Expired / revoked token (e.g. after changing the password elsewhere): return to the login screen.
  const expireSession = useCallback(() => {
    setAuthToken(null);
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
    setUserEmail(null);
    setProfile(null);
    setDocumentId(null);
    setUploads([]);
    setConversations([]);
    setConversationId(null);
    setInitialMessages([]);
    setChatKey((k) => k + 1);
    setResetToken(null);
    setLoginNotice("Your session has expired. Please log in again.");
    setAuthScreen(AUTH_SCREEN.LOGIN);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(expireSession);
    return () => setUnauthorizedHandler(null);
  }, [expireSession]);

  // Fetch the real profile (name, email, etc.) after login. The /auth/* endpoint is outside the shared 401 interceptor,
  // so handle a 401 here separately: the token is no longer valid -> return to the login screen.
  useEffect(() => {
    if (!userEmail) return undefined;
    let cancelled = false;
    getMe()
      .then((res) => {
        if (!cancelled) setProfile(res.data);
      })
      .catch((err) => {
        if (!cancelled && err?.response?.status === 401) expireSession();
      });
    return () => {
      cancelled = true;
    };
  }, [userEmail, expireSession]);

  // A password change revokes the old token: accept the new token and preserve "Remember me" if enabled.
  const handlePasswordChanged = (token) => {
    setAuthToken(token);
    if (localStorage.getItem("auth_token")) localStorage.setItem("auth_token", token);
  };

  const handleAuthenticated = (email, token) => {
    if (token) setAuthToken(token);
    setUserEmail(email);
  };

  const goLogin = (notice = null) => {
    setLoginNotice(notice);
    setResetToken(null);
    setAuthScreen(AUTH_SCREEN.LOGIN);
  };

  const goSignup = () => {
    setResetToken(null);
    setAuthScreen(AUTH_SCREEN.SIGNUP);
  };

  const handleResetSuccess = () => {
    setAuthToken(null);
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
    goLogin("Your password has been updated. Please log in with your new password.");
  };

  const handleLogout = () => {
    setAuthToken(null);
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
    setUserEmail(null);
    setProfile(null);
    setDocumentId(null);
    setUploads([]);
    setConversations([]);
    setConversationId(null);
    setInitialMessages([]);
    setChatKey((k) => k + 1);
    setNavOpen(false);
    setConfirmingLogout(false);
    setPendingDelete(null);
    goLogin();
  };

  const updateUpload = (key, patch) =>
    setUploads((prev) => prev.map((u) => (u.key === key ? { ...u, ...patch } : u)));

  const handleFilesChange = async (event) => {
    const files = Array.from(event.target.files || []).filter((f) =>
      ALLOWED_EXT.test(f.name)
    );
    event.target.value = ""; 
    if (!files.length) return;

    const stamp = Date.now();
    const items = files.map((file, i) => ({
      key: `${stamp}-${i}-${file.name}`,
      file,
      name: file.name,
    }));
    setUploads((prev) => [
      ...items.map(({ key, name }) => ({ key, name, id: null, status: "uploading", error: null })),
      ...prev,
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
            error: getErrorMessage(err, "Failed to upload file."),
          });
        }
      }
    };
    await Promise.all(
      Array.from({ length: Math.min(UPLOAD_CONCURRENCY, items.length) }, worker)
    );
  };

  const refreshConversations = useCallback(async () => {
    try {
      const res = await listConversations();
      setConversations(res.data);
    } catch {
      /* The list is only a convenience; the next action will retry. */
    }
  }, []);

  // After login (or a page reload), restore documents and conversations saved on the server.
  useEffect(() => {
    if (!userEmail) return undefined;
    let cancelled = false;
    listDocuments()
      .then((res) => {
        if (!cancelled) setUploads((prev) => mergeServerDocs(prev, res.data));
      })
      .catch(() => {
        /* Keep the current list unchanged. */
      });
    refreshConversations();
    return () => {
      cancelled = true;
    };
  }, [userEmail, refreshConversations]);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = setTimeout(() => setNotice(null), 6000);
    return () => clearTimeout(timer);
  }, [notice]);

  const startNewChat = () => {
    setConversationId(null);
    setInitialMessages([]);
    setChatKey((k) => k + 1);
    navigate("chat");
  };

  const selectConversation = async (id) => {
    navigate("chat");
    if (id === conversationId) return;
    try {
      const res = await getConversation(id);
      setInitialMessages(res.data.messages.map(toUiMessage));
      setConversationId(id);
      setDocumentId(res.data.document_id || null);
      setChatKey((k) => k + 1);
    } catch (err) {
      setNotice(getErrorMessage(err, "Không mở được cuộc trò chuyện này."));
      refreshConversations();
    }
  };

  const requestDeleteDocument = (u) => {
    if (!u.id) {
      // Item not yet on the server (upload failed): just remove it from the list.
      setUploads((prev) => prev.filter((x) => x.key !== u.key));
      return;
    }
    setPendingDelete({ type: "document", id: u.id, name: u.name });
  };

  const requestDeleteConversation = (c) =>
    setPendingDelete({ type: "conversation", id: c.id, name: c.title });

  const confirmDelete = async () => {
    const target = pendingDelete;
    setPendingDelete(null);
    if (!target) return;
    try {
      if (target.type === "document") {
        await deleteDocument(target.id);
        setUploads((prev) => prev.filter((u) => u.id !== target.id));
        if (documentId === target.id) setDocumentId(null);
      } else {
        await deleteConversation(target.id);
        setConversations((prev) => prev.filter((c) => c.id !== target.id));
        if (conversationId === target.id) startNewChat();
      }
    } catch (err) {
      setNotice(getErrorMessage(err, "Không xóa được. Vui lòng thử lại."));
    }
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


  if (!userEmail) {
    switch (authScreen) {
      case AUTH_SCREEN.SIGNUP:
        return (
          <SignupPage
            onGoLogin={(notice, email) => {
              if (email) setResetEmail(email);
              goLogin(notice);
            }}
          />
        );

      case AUTH_SCREEN.FORGOT:
        return (
          <ForgotPasswordPage
            initialEmail={resetEmail}
            onCodeSent={(email) => {
              setResetEmail(email);
              setAuthScreen(AUTH_SCREEN.VERIFY);
            }}
            onGoLogin={() => goLogin()}
            onGoSignup={goSignup}
          />
        );

      case AUTH_SCREEN.VERIFY:
        if (!resetEmail) return <Navigate to={AUTH_SCREEN.FORGOT} onNavigate={setAuthScreen} />;
        return (
          <VerifyCodePage
            email={resetEmail}
            onVerified={(token) => {
              setResetToken(token);
              setAuthScreen(AUTH_SCREEN.RESET);
            }}
            onChangeEmail={() => setAuthScreen(AUTH_SCREEN.FORGOT)}
            onGoLogin={() => goLogin()}
            onGoSignup={goSignup}
          />
        );

      case AUTH_SCREEN.RESET:
        if (!resetToken) return <Navigate to={AUTH_SCREEN.FORGOT} onNavigate={setAuthScreen} />;
        return (
          <ResetPasswordPage
            resetToken={resetToken}
            onSuccess={handleResetSuccess}
            onSessionExpired={() => {
              setResetToken(null);
              setAuthScreen(AUTH_SCREEN.FORGOT);
            }}
            onGoLogin={() => goLogin()}
            onGoSignup={goSignup}
          />
        );

      default:
        return (
          <LoginPage
            key={loginNotice || "login"}
            onAuthenticated={handleAuthenticated}
            onGoSignup={goSignup}
            onGoForgot={() => {
              setLoginNotice(null);
              setAuthScreen(AUTH_SCREEN.FORGOT);
            }}
            notice={loginNotice}
            initialIdentifier={loginNotice ? resetEmail : ""}
          />
        );
    }
  }

  const scopeName = documentId ? uploads.find((u) => u.id === documentId)?.name : null;
  const scopeLabel = scopeName || "Tất cả tài liệu";

  return (
    <div className="app" data-nav={navOpen ? "open" : "closed"}>
      <Sidebar
        view={view}
        onNavigate={navigate}
        conversations={conversations}
        conversationId={conversationId}
        onNewChat={() => {
          startNewChat();
          setNavOpen(false);
        }}
        onSelectConversation={(id) => {
          selectConversation(id);
          setNavOpen(false);
        }}
        onDeleteConversation={requestDeleteConversation}
        uploads={uploads}
        documentId={documentId}
        onSelectDocument={(id) => {
          setDocumentId(id);
          setNavOpen(false);
        }}
        onDeleteDocument={requestDeleteDocument}
        onFiles={handleFilesChange}
        profile={profile}
        fallbackName={userEmail}
        onLogout={() => setConfirmingLogout(true)}
        onClose={() => setNavOpen(false)}
      />
      <div className="scrim" onClick={() => setNavOpen(false)} />

      <div className="main">
        <div className="topbar">
          <button
            type="button"
            className="icon-btn"
            aria-label="Mở thanh bên"
            onClick={() => setNavOpen(true)}
          >
            <Icon name="menu" size={20} />
          </button>
          <span className="brand">AI Study Assistant</span>
        </div>

        {/* Keep Q&A / Quiz / Flashcards in memory so their content is preserved when navigating between pages. */}
        <main className="view view--chat" hidden={view !== "chat"}>
          <ChatWindow
            key={chatKey}
            documentId={documentId}
            scopeLabel={scopeLabel}
            hasDocuments={uploads.length > 0}
            conversationId={conversationId}
            initialMessages={initialMessages}
            onConversationCreated={setConversationId}
            onActivity={refreshConversations}
          />
        </main>
        <main className="view" hidden={view !== "quiz"}>
          <QuizPlayer documentId={documentId} scopeLabel={scopeLabel} />
        </main>
        <main className="view" hidden={view !== "flashcard"}>
          <FlashcardDeck documentId={documentId} scopeLabel={scopeLabel} />
        </main>
        {view === "profile" && (
          <main className="view">
            <ProfilePage profile={profile} onUpdated={setProfile} />
          </main>
        )}
        {view === "settings" && (
          <main className="view">
            <SettingsPage
              profile={profile}
              onPasswordChanged={handlePasswordChanged}
              onLogout={() => setConfirmingLogout(true)}
            />
          </main>
        )}
      </div>

      {confirmingLogout && (
        <ConfirmDialog
          title="Đăng xuất?"
          message="Bạn sẽ cần đăng nhập lại để tiếp tục sử dụng. Tài liệu và các cuộc trò chuyện của bạn vẫn được lưu."
          confirmLabel="Đăng xuất"
          danger
          onConfirm={handleLogout}
          onCancel={() => setConfirmingLogout(false)}
        />
      )}

      {pendingDelete && (
        <ConfirmDialog
          title={pendingDelete.type === "document" ? "Xóa tài liệu?" : "Xóa cuộc trò chuyện?"}
          message={
            pendingDelete.type === "document"
              ? `“${pendingDelete.name}” sẽ bị xóa cùng toàn bộ dữ liệu đã xử lý. Không thể hoàn tác. Lịch sử trò chuyện và quiz vẫn được giữ.`
              : `Toàn bộ tin nhắn trong “${pendingDelete.name}” sẽ bị xóa. Không thể hoàn tác.`
          }
          confirmLabel="Xóa"
          danger
          onConfirm={confirmDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}

      {notice && (
        <div className="toast" role="alert">
          {notice}
        </div>
      )}
    </div>
  );
}
