import { useState } from "react";
import { changePassword, getErrorMessage } from "../../services/api";
import { getThemePreference, setThemePreference } from "../../theme";

const THEMES = [
  { key: "light", label: "Sáng" },
  { key: "dark", label: "Tối" },
  { key: "system", label: "Theo hệ thống" },
];

const MAX_PASSWORD_BYTES = 72; // giới hạn của bcrypt, khớp với backend

const EMPTY_FORM = { current: "", next: "", confirm: "" };

export default function SettingsPage({ onPasswordChanged, onLogout }) {
  const [theme, setTheme] = useState(getThemePreference);
  const [pwOpen, setPwOpen] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);

  const chooseTheme = (key) => {
    setThemePreference(key);
    setTheme(key);
  };

  const setField = (field) => (e) => setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const closeForm = () => {
    setPwOpen(false);
    setForm(EMPTY_FORM);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (saving) return;
    setDone(false);

    if (!form.current || !form.next) {
      setError("Vui lòng nhập mật khẩu hiện tại và mật khẩu mới.");
      return;
    }
    if (form.next.length < 8) {
      setError("Mật khẩu mới cần ít nhất 8 ký tự.");
      return;
    }
    if (new TextEncoder().encode(form.next).length > MAX_PASSWORD_BYTES) {
      setError("Mật khẩu mới quá dài (tối đa 72 byte).");
      return;
    }
    if (form.next !== form.confirm) {
      setError("Mật khẩu xác nhận không khớp.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      const res = await changePassword(form.current, form.next, form.confirm);
      onPasswordChanged(res.data.access_token);
      setDone(true);
      closeForm();
    } catch (err) {
      setError(getErrorMessage(err, "Không thể đổi mật khẩu. Vui lòng thử lại."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page">
      <header className="page-head">
        <h1 className="page-title">Cài đặt</h1>
        <p className="page-sub">Quản lý tài khoản và giao diện.</p>
      </header>

      <section className="card card-pad">
        <h2 className="card-title">Tài khoản</h2>
        <p className="card-sub">Thiết lập đăng nhập và bảo mật.</p>

        <div style={{ marginTop: 12 }}>
          <div className="setting-row">
            <div>
              <div className="setting-name">Trạng thái tài khoản</div>
              <div className="setting-desc">Tài khoản của bạn đang hoạt động.</div>
            </div>
            <span className="badge badge--ok">Đang hoạt động</span>
          </div>

          <div className="setting-row">
            <div>
              <div className="setting-name">Mật khẩu</div>
              <div className="setting-desc">Đổi mật khẩu đăng nhập của bạn.</div>
            </div>
            {!pwOpen && (
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setDone(false);
                  setPwOpen(true);
                }}
              >
                Đổi mật khẩu
              </button>
            )}
          </div>

          {done && !pwOpen && (
            <p className="note note--ok" role="status" style={{ marginBottom: 16 }}>
              Đã đổi mật khẩu. Các thiết bị khác sẽ cần đăng nhập lại.
            </p>
          )}

          {pwOpen && (
            <form className="pw-form" onSubmit={handleSubmit} noValidate>
              <div className="field">
                <label className="label" htmlFor="pw-current">
                  Mật khẩu hiện tại
                </label>
                <input
                  id="pw-current"
                  className="input"
                  type="password"
                  value={form.current}
                  onChange={setField("current")}
                  autoComplete="current-password"
                  autoFocus
                />
              </div>
              <div className="form-grid">
                <div className="field">
                  <label className="label" htmlFor="pw-new">
                    Mật khẩu mới
                  </label>
                  <input
                    id="pw-new"
                    className="input"
                    type="password"
                    value={form.next}
                    onChange={setField("next")}
                    autoComplete="new-password"
                  />
                  <span className="field-hint">Ít nhất 8 ký tự.</span>
                </div>
                <div className="field">
                  <label className="label" htmlFor="pw-confirm">
                    Nhập lại mật khẩu mới
                  </label>
                  <input
                    id="pw-confirm"
                    className="input"
                    type="password"
                    value={form.confirm}
                    onChange={setField("confirm")}
                    autoComplete="new-password"
                  />
                </div>
              </div>
              {error && (
                <p className="note note--err" role="alert">
                  {error}
                </p>
              )}
              <div className="actions">
                <button type="submit" className="btn btn--primary" disabled={saving}>
                  {saving ? "Đang lưu…" : "Cập nhật mật khẩu"}
                </button>
                <button type="button" className="btn" onClick={closeForm} disabled={saving}>
                  Hủy
                </button>
              </div>
            </form>
          )}

          <div className="setting-row">
            <div>
              <div className="setting-name" style={{ color: "var(--danger)" }}>
                Đăng xuất
              </div>
              <div className="setting-desc">Đăng xuất khỏi tài khoản trên thiết bị này.</div>
            </div>
            <button type="button" className="btn btn--danger" onClick={onLogout}>
              Đăng xuất
            </button>
          </div>
        </div>
      </section>

      <section className="card card-pad">
        <div className="setting-row" style={{ padding: 0 }}>
          <div>
            <h2 className="card-title">Giao diện</h2>
            <p className="card-sub">Chọn màu nền cho ứng dụng.</p>
          </div>
          <div className="segmented" role="group" aria-label="Giao diện">
            {THEMES.map((t) => (
              <button key={t.key} type="button" aria-pressed={theme === t.key} onClick={() => chooseTheme(t.key)}>
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
