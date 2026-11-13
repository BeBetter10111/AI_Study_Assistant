import { useEffect, useRef, useState } from "react";
import Icon from "./Icons";
import { displayName, initials } from "../utils/user";

const NAV = [
  { key: "chat", label: "Hỏi đáp", icon: "chat" },
  { key: "quiz", label: "Quiz", icon: "quiz" },
  { key: "flashcard", label: "Thẻ ghi nhớ", icon: "cards" },
];

const STATUS_TEXT = {
  uploading: "Đang tải lên…",
  pending: "Đang xử lý…",
  failed: "Không xử lý được",
};

export default function Sidebar({
  view,
  onNavigate,
  uploads,
  documentId,
  onSelectDocument,
  onFiles,
  profile,
  fallbackName,
  onLogout,
  onClose,
}) {
  const fileInput = useRef(null);
  const folderInput = useRef(null);
  const footRef = useRef(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (!menuOpen) return undefined;
    const onPointer = (e) => {
      if (footRef.current && !footRef.current.contains(e.target)) setMenuOpen(false);
    };
    const onKey = (e) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  const name = displayName(profile, fallbackName);
  const go = (key) => {
    setMenuOpen(false);
    onNavigate(key);
  };

  return (
    <aside className="sidebar" aria-label="Thanh bên">
      <div className="sb-head">
        <span className="brand">AI Study Assistant</span>
        <button type="button" className="icon-btn sb-close" aria-label="Đóng thanh bên" onClick={onClose}>
          <Icon name="close" />
        </button>
      </div>

      <nav className="nav" aria-label="Chức năng">
        {NAV.map((item) => (
          <button
            key={item.key}
            type="button"
            className="nav-item"
            aria-current={view === item.key ? "page" : undefined}
            onClick={() => go(item.key)}
          >
            <Icon name={item.icon} />
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sb-section">
        <span>Tài liệu</span>
        <button
          type="button"
          className="icon-btn"
          aria-label="Tải tài liệu lên"
          title="Tải tài liệu lên"
          onClick={() => fileInput.current?.click()}
        >
          <Icon name="plus" size={16} />
        </button>
      </div>

      <div className="doc-list">
        {uploads.length === 0 ? (
          <button type="button" className="sb-empty" onClick={() => fileInput.current?.click()}>
            Tải lên PDF, DOCX, PPTX hoặc Markdown để hỏi đáp và tạo quiz theo nội dung.
          </button>
        ) : (
          <>
            <button
              type="button"
              className="doc"
              aria-pressed={documentId === null}
              onClick={() => onSelectDocument(null)}
            >
              <Icon name="folder" size={16} />
              <span className="doc-body">
                <span className="doc-name">Tất cả tài liệu</span>
              </span>
            </button>
            {uploads.map((u) => {
              const ready = u.status === "embedded";
              const busy = u.status === "uploading" || u.status === "pending";
              return (
                <button
                  key={u.key}
                  type="button"
                  className="doc"
                  aria-pressed={ready && documentId === u.id}
                  disabled={!ready}
                  title={u.name}
                  onClick={() => onSelectDocument(u.id)}
                >
                  {busy ? <span className="spinner" style={{ marginTop: 4 }} /> : <Icon name="file" size={16} />}
                  <span className="doc-body">
                    <span className="doc-name">{u.name}</span>
                    {!ready && (
                      <span className={`doc-meta${u.status === "failed" ? " doc-meta--err" : ""}`}>
                        {u.status === "failed" && u.error ? u.error : STATUS_TEXT[u.status] || u.status}
                      </span>
                    )}
                  </span>
                </button>
              );
            })}
          </>
        )}
        <button type="button" className="sb-link" onClick={() => folderInput.current?.click()}>
          Tải cả thư mục
        </button>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept=".pdf,.docx,.pptx,.md"
          onChange={onFiles}
          hidden
        />
        <input ref={folderInput} type="file" multiple webkitdirectory="" onChange={onFiles} hidden />
      </div>

      <div className="sb-foot" ref={footRef}>
        {menuOpen && (
          <div className="menu" role="menu">
            {profile?.email && <div className="menu-email">{profile.email}</div>}
            <button type="button" role="menuitem" className="menu-item" onClick={() => go("profile")}>
              <Icon name="user" /> Hồ sơ cá nhân
            </button>
            <button type="button" role="menuitem" className="menu-item" onClick={() => go("settings")}>
              <Icon name="sliders" /> Cài đặt
            </button>
            <button type="button" role="menuitem" className="menu-item"
              onClick={() => {
                setMenuOpen(false);
                onLogout();
              }}
            >
              <Icon name="logout" /> Đăng xuất
            </button>
          </div>
        )}
        <button
          type="button"
          className="user-btn"
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span className="avatar">{initials(name)}</span>
          <span className="user-text">
            <span className="user-name">{name}</span>
            {profile?.username && profile.full_name && <span className="user-sub">@{profile.username}</span>}
          </span>
          <Icon name="updown" size={16} />
        </button>
      </div>
    </aside>
  );
}
