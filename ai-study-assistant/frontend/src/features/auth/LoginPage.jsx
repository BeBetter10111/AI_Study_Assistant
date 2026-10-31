import { useState } from "react";
import { loginUser, setAuthToken } from "../../services/api";

export default function LoginPage({
  onAuthenticated,
  onGoSignup,
  onGoForgot,
  notice = null,
  initialIdentifier = "",
}) {
  const [identifier, setIdentifier] = useState(initialIdentifier);
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!identifier.trim() || !password.trim() || isLoading) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await loginUser(identifier.trim(), password);
      const { access_token: token } = response.data;
      setAuthToken(token);
      if (rememberMe) {
        localStorage.setItem("auth_token", token);
        localStorage.setItem("auth_email", identifier.trim());
      }
      onAuthenticated(identifier.trim(), token);
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed. Please check your credentials.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.appTitle}>AI Study Assistant</h1>

      <div style={styles.card}>
        <h2 style={styles.cardTitle}>Login to your account</h2>
        <p style={styles.cardSubtitle}>
          Enter your username or email below to login to your account
        </p>

        {notice && <p style={styles.notice}>{notice}</p>}

        <form onSubmit={handleLogin} style={styles.form}>
          <div style={styles.fieldGroup}>
            <label style={styles.label}>Username or Email</label>
            <input
              style={styles.input}
              type="text"
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              autoComplete="username"
              disabled={isLoading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <div style={styles.passwordHeader}>
              <label style={styles.label}>Password</label>
              <button type="button" style={styles.forgotLink} onClick={onGoForgot}>
                Forgot your password?
              </button>
            </div>
            <input
              style={styles.input}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              disabled={isLoading}
            />
          </div>

          <label style={styles.checkboxRow}>
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              style={{ marginRight: "6px" }}
            />
            Remember Me
          </label>

          {error && <p style={styles.error}>{error}</p>}

          <button type="submit" style={styles.submitButton} disabled={isLoading}>
            {isLoading ? "Logging in…" : "Login"}
          </button>
        </form>

        <p style={styles.switchText}>
          Don&apos;t have an account?{" "}
          <button type="button" style={styles.switchLink} onClick={onGoSignup}>
            Sign up
          </button>
        </p>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    backgroundColor: "#f8f9fa",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "system-ui, -apple-system, sans-serif",
    padding: "24px",
    boxSizing: "border-box",
  },
  appTitle: {
    fontSize: "28px",
    fontWeight: "700",
    color: "#111827",
    marginBottom: "32px",
    textAlign: "center",
  },
  card: {
    backgroundColor: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "12px",
    padding: "32px",
    width: "100%",
    maxWidth: "380px",
    boxSizing: "border-box",
  },
  cardTitle: {
    fontSize: "20px",
    fontWeight: "600",
    color: "#111827",
    margin: "0 0 4px 0",
  },
  cardSubtitle: {
    fontSize: "13px",
    color: "#6b7280",
    margin: "0 0 24px 0",
    lineHeight: "1.5",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  fieldGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  label: {
    fontSize: "14px",
    fontWeight: "500",
    color: "#374151",
  },
  passwordHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  forgotLink: {
    fontSize: "13px",
    color: "#2563eb",
    background: "none",
    border: "none",
    cursor: "pointer",
    textDecoration: "underline",
    padding: 0,
  },
  input: {
    padding: "8px 12px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "14px",
    outline: "none",
    width: "100%",
    boxSizing: "border-box",
    color: "#111827",
    backgroundColor: "#fff",
  },
  checkboxRow: {
    display: "flex",
    alignItems: "center",
    fontSize: "14px",
    color: "#374151",
    cursor: "pointer",
  },
  submitButton: {
    width: "100%",
    padding: "10px",
    backgroundColor: "#111827",
    color: "#ffffff",
    border: "none",
    borderRadius: "6px",
    fontSize: "15px",
    fontWeight: "600",
    cursor: "pointer",
    marginTop: "4px",
  },
  error: {
    fontSize: "13px",
    color: "#dc2626",
    margin: 0,
  },
  notice: {
    fontSize: "13px",
    color: "#166534",
    backgroundColor: "#f0fdf4",
    border: "1px solid #bbf7d0",
    borderRadius: "6px",
    padding: "8px 12px",
    margin: "0 0 16px 0",
  },
  switchText: {
    textAlign: "center",
    fontSize: "14px",
    color: "#6b7280",
    marginTop: "20px",
    marginBottom: 0,
  },
  switchLink: {
    color: "#2563eb",
    background: "none",
    border: "none",
    cursor: "pointer",
    textDecoration: "underline",
    fontSize: "14px",
    padding: 0,
  },
};
