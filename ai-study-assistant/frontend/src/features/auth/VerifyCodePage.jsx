import { useEffect, useState } from "react";
import { forgotPassword, verifyResetCode } from "../../services/api";
import { authStyles as styles } from "./authStyles";
import AuthFooterLinks from "./AuthFooterLinks";

const RESEND_COOLDOWN_S = 60; // khớp PASSWORD_RESET_RESEND_COOLDOWN_S ở backend

// Bước 2: user mở email, copy mã UUID và dán vào đây.
export default function VerifyCodePage({ email, onVerified, onChangeEmail, onGoLogin, onGoSignup }) {
  const [code, setCode] = useState("");
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [cooldown, setCooldown] = useState(RESEND_COOLDOWN_S);

  useEffect(() => {
    if (cooldown <= 0) return undefined;
    const timer = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  const handleVerify = async (e) => {
    e.preventDefault();
    const value = code.trim();
    if (!value || isLoading) return;
    setIsLoading(true);
    setError(null);
    setInfo(null);
    try {
      const res = await verifyResetCode(email, value);
      onVerified(res.data.reset_token);
    } catch (err) {
      setError(err.response?.data?.detail || "Verification failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleResend = async () => {
    if (cooldown > 0) return;
    setError(null);
    setInfo(null);
    try {
      await forgotPassword(email);
      setCooldown(RESEND_COOLDOWN_S);
      setCode("");
      setInfo("A new code has been sent. Previous codes are no longer valid.");
    } catch (err) {
      setError(err.response?.data?.detail || "Could not resend the code.");
    }
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.appTitle}>AI Study Assistant</h1>
      <div style={styles.card}>
        <h2 style={styles.cardTitle}>Verify your UUID</h2>
        <p style={styles.cardSubtitle}>
          Enter the UUID we sent to <strong>{email}</strong> to verify. Check your spam folder if you
          don&apos;t see it.
        </p>

        <form onSubmit={handleVerify} style={styles.form}>
          <div style={styles.fieldGroup}>
            <label style={styles.label}>UUID</label>
            <input
              style={styles.input}
              type="text"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
              autoComplete="one-time-code"
              spellCheck={false}
              disabled={isLoading}
              autoFocus
            />
          </div>

          {error && <p style={styles.error}>{error}</p>}
          {info && <p style={styles.info}>{info}</p>}

          <button type="submit" style={styles.submitButton} disabled={isLoading}>
            {isLoading ? "Verifying…" : "Verify"}
          </button>

          <button
            type="button"
            style={{ ...styles.linkButton, opacity: cooldown > 0 ? 0.5 : 1 }}
            onClick={handleResend}
            disabled={cooldown > 0}
          >
            {cooldown > 0 ? `Resend code in ${cooldown}s` : "Resend code"}
          </button>
          <button type="button" style={styles.linkButton} onClick={onChangeEmail}>
            Use a different email
          </button>
        </form>

        <AuthFooterLinks onGoLogin={onGoLogin} onGoSignup={onGoSignup} />
      </div>
    </div>
  );
}
