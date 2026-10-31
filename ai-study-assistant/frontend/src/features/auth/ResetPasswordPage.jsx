import { useState } from "react";
import { resetPassword } from "../../services/api";
import { authStyles as styles } from "./authStyles";
import AuthFooterLinks from "./AuthFooterLinks";

const MIN_LENGTH = 8;
const MAX_BYTES = 72; // giới hạn của bcrypt, khớp backend

// Bước 3: đặt mật khẩu mới (chỉ vào được sau khi mã đã được xác nhận).
export default function ResetPasswordPage({ resetToken, onSuccess, onSessionExpired, onGoLogin, onGoSignup }) {
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isLoading) return;
    setError(null);

    if (password.length < MIN_LENGTH) {
      setError(`Password must be at least ${MIN_LENGTH} characters.`);
      return;
    }
    if (new TextEncoder().encode(password).length > MAX_BYTES) {
      setError(`Password must be at most ${MAX_BYTES} bytes.`);
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }

    setIsLoading(true);
    try {
      await resetPassword(resetToken, password, confirm);
      onSuccess();
    } catch (err) {
      const detail = err.response?.data?.detail;
      const message = Array.isArray(detail) ? detail[0]?.msg : detail;
      // Phiên reset hết hạn / mã đã dùng -> quay lại bước 1.
      if (err.response?.status === 400 && /session/i.test(message || "")) {
        onSessionExpired(message);
        return;
      }
      setError(message || "Could not reset password. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.appTitle}>AI Study Assistant</h1>
      <div style={styles.card}>
        <h2 style={styles.cardTitle}>Resetting your password</h2>
        <p style={styles.cardSubtitle}>Enter your new password below to proceed</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.fieldGroup}>
            <label style={styles.label}>Password</label>
            <input
              style={styles.input}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
              disabled={isLoading}
              autoFocus
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Confirm Password</label>
            <input
              style={styles.input}
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              autoComplete="new-password"
              disabled={isLoading}
            />
          </div>

          {error && <p style={styles.error}>{error}</p>}

          <button type="submit" style={styles.submitButton} disabled={isLoading}>
            {isLoading ? "Updating…" : "Reset Password"}
          </button>
        </form>

        <AuthFooterLinks onGoLogin={onGoLogin} onGoSignup={onGoSignup} />
      </div>
    </div>
  );
}
