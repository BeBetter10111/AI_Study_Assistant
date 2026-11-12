import { useCallback, useEffect, useState } from "react";
import {
  getErrorMessage,
  getMe,
  setAuthToken,
  setUnauthorizedHandler,
  uploadDocument,
  getDocumentStatus,
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
import Icon from "./components/Icons";

const UPLOAD_CONCURRENCY = 3;
const ALLOWED_EXT = /\.(pdf|docx|pptx|md)$/i;
const DONE_STATUSES = ["embedded", "failed"];

// Các màn hình sau đăng nhập. Lưu trong URL hash (#/profile) để tải lại trang vẫn ở đúng chỗ.
const VIEWS = ["chat", "quiz", "flashcard", "profile", "settings"];
const readView = () => {
  const name = window.location.hash.replace(/^#\/?/, "");
  return VIEWS.includes(name) ? name : "chat";
};

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

  // Token hết hạn / bị vô hiệu hoá (ví dụ đã đổi mật khẩu ở nơi khác): quay về màn hình đăng nhập.
  const expireSession = useCallback(() => {
    setAuthToken(null);
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
    setUserEmail(null);
    setProfile(null);
    setDocumentId(null);
    setUploads([]);
    setResetToken(null);
    setLoginNotice("Your session has expired. Please log in again.");
    setAuthScreen(AUTH_SCREEN.LOGIN);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(expireSession);
    return () => setUnauthorizedHandler(null);
  }, [expireSession]);

  // Lấy hồ sơ thật (họ tên, email...) sau khi đăng nhập. Endpoint /auth/* nằm ngoài interceptor 401
  // chung, nên 401 ở đây được xử lý riêng: token không còn hợp lệ -> về màn hình đăng nhập.
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

  // Đổi mật khẩu thu hồi token cũ: nhận token mới và giữ nguyên chế độ "Remember me" nếu đang bật.
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
    setNavOpen(false);
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
            error: getErrorMessage(err, "Failed to upload file."),
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
        uploads={uploads}
        documentId={documentId}
        onSelectDocument={(id) => {
          setDocumentId(id);
          setNavOpen(false);
        }}
        onFiles={handleFilesChange}
        profile={profile}
        fallbackName={userEmail}
        onLogout={handleLogout}
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

        {/* Hỏi đáp / Quiz / Thẻ ghi nhớ luôn được giữ trong bộ nhớ để chuyển trang không mất nội dung. */}
        <main className="view view--chat" hidden={view !== "chat"}>
          <ChatWindow documentId={documentId} scopeLabel={scopeLabel} hasDocuments={uploads.length > 0} />
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
              onLogout={handleLogout}
            />
          </main>
        )}
      </div>
    </div>
  );
}
