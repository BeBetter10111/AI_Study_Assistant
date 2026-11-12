import { useState } from "react";
import { getErrorMessage, loginUser, setAuthToken } from "../../services/api";
import { authStyles as styles } from "./authStyles";

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
      setError(getErrorMessage(err, "Login failed. Please check your credentials."));
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
