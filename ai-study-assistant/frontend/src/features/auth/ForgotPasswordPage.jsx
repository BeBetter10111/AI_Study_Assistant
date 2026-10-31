import { useState } from "react";
import { forgotPassword } from "../../services/api";
import { authStyles as styles } from "./authStyles";
import AuthFooterLinks from "./AuthFooterLinks";

// Bước 1: nhập email -> backend gửi mã UUID qua email.
export default function ForgotPasswordPage({ initialEmail = "", onCodeSent, onGoLogin, onGoSignup }) {
  const [email, setEmail] = useState(initialEmail);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const value = email.trim();
    if (!value || isLoading) return;
    setIsLoading(true);
    setError(null);
    try {
      await forgotPassword(value);
      // Backend luôn trả thành công (không lộ email nào tồn tại) -> sang bước nhập mã.
      onCodeSent(value);
    } catch (err) {
      setError(err.response?.data?.detail?.[0]?.msg || err.response?.data?.detail || "Could not send the code. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.appTitle}>AI Study Assistant</h1>
      <div style={styles.card}>
        <h2 style={styles.cardTitle}>Resetting your password</h2>
        <p style={styles.cardSubtitle}>Enter your email below to reset your account&apos;s password</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.fieldGroup}>
            <label style={styles.label}>Email</label>
            <input
              style={styles.input}
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              disabled={isLoading}
              autoFocus
            />
          </div>

          {error && <p style={styles.error}>{typeof error === "string" ? error : "Invalid email."}</p>}

          <button type="submit" style={styles.submitButton} disabled={isLoading}>
            {isLoading ? "Sending…" : "Reset Password"}
          </button>
        </form>

        <AuthFooterLinks onGoLogin={onGoLogin} onGoSignup={onGoSignup} />
      </div>
    </div>
  );
}
