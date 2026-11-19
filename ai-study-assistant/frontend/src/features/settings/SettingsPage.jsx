import { useState } from "react";
import { changePassword, getErrorMessage } from "../../services/api";
import { getThemePreference, setThemePreference } from "../../theme";

const THEMES = [
  { key: "light", label: "Light" },
  { key: "dark", label: "Dark" },
  { key: "system", label: "System" },
];

const MAX_PASSWORD_BYTES = 72; // bcrypt limit, matching the backend.

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
      setError("Please enter your current password and new password.");
      return;
    }
    if (form.next.length < 8) {
      setError("New password must be at least 8 characters.");
      return;
    }
    if (new TextEncoder().encode(form.next).length > MAX_PASSWORD_BYTES) {
      setError("New password is too long (maximum 72 bytes).");
      return;
    }
    if (form.next !== form.confirm) {
      setError("Passwords do not match.");
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
      setError(getErrorMessage(err, "Could not change password. Please try again."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page">
      <header className="page-head">
        <h1 className="page-title">Settings</h1>
        <p className="page-sub">Manage your account and appearance.</p>
      </header>

      <section className="card card-pad">
        <h2 className="card-title">Account</h2>
        <p className="card-sub">Manage sign-in and security settings.</p>

        <div style={{ marginTop: 12 }}>
          <div className="setting-row">
            <div>
              <div className="setting-name">Account status</div>
              <div className="setting-desc">Your account is active.</div>
            </div>
            <span className="badge badge--ok">Active</span>
          </div>

          <div className="setting-row">
            <div>
              <div className="setting-name">Password</div>
              <div className="setting-desc">Change your sign-in password.</div>
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
                Change password
              </button>
            )}
          </div>

          {done && !pwOpen && (
            <p className="note note--ok" role="status" style={{ marginBottom: 16 }}>
              Password changed. Other devices will need to sign in again.
            </p>
          )}

          {pwOpen && (
            <form className="pw-form" onSubmit={handleSubmit} noValidate>
              <div className="field">
                <label className="label" htmlFor="pw-current">
                  Current password
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
                    New password
                  </label>
                  <input
                    id="pw-new"
                    className="input"
                    type="password"
                    value={form.next}
                    onChange={setField("next")}
                    autoComplete="new-password"
                  />
                  <span className="field-hint">At least 8 characters.</span>
                </div>
                <div className="field">
                  <label className="label" htmlFor="pw-confirm">
                    Confirm new password
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
                  {saving ? "Saving…" : "Update password"}
                </button>
                <button type="button" className="btn" onClick={closeForm} disabled={saving}>
                  Cancel
                </button>
              </div>
            </form>
          )}

          <div className="setting-row">
            <div>
              <div className="setting-name" style={{ color: "var(--danger)" }}>
                Log out
              </div>
              <div className="setting-desc">Log out of your account on this device.</div>
            </div>
            <button type="button" className="btn btn--danger" onClick={onLogout}>
              Log out
            </button>
          </div>
        </div>
      </section>

      <section className="card card-pad">
        <div className="setting-row" style={{ padding: 0 }}>
          <div>
            <h2 className="card-title">Appearance</h2>
            <p className="card-sub">Choose the application theme.</p>
          </div>
          <div className="segmented" role="group" aria-label="Appearance">
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
