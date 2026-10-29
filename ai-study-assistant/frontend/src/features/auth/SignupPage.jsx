import { useState } from "react";
import { registerUser, loginUser, setAuthToken } from "../../services/api";

const ROLES = ["Admin", "User"];

export default function SignupPage({ onAuthenticated, onGoLogin }) {
  const [form, setForm] = useState({
    username: "",
    fullName: "",
    email: "",
    phone: "",
    password: "",
    confirmPassword: "",
    role: "",
  });
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (field) => (e) =>
    setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSignup = async (e) => {
    e.preventDefault();
    setError(null);

    if (!form.username.trim() || !form.email.trim() || !form.password) {
      setError("Please fill in all required fields.");
      return;
    }
    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (form.password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    setIsLoading(true);
    try {
      await registerUser(form.email.trim(), form.password);
      // Auto-login after successful registration
      const loginRes = await loginUser(form.email.trim(), form.password);
      const { access_token: token } = loginRes.data;
      setAuthToken(token);
      onAuthenticated(form.email.trim(), token);
    } catch (err) {
      setError(err.response?.data?.detail || "Sign up failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.appTitle}>AI Study Assistant</h1>

      <div style={styles.card}>
        <h2 style={styles.cardTitle}>Sign up for a new account</h2>
        <p style={styles.cardSubtitle}>Enter your information below to sign up</p>

        <form onSubmit={handleSignup} style={styles.form}>
          <div style={styles.fieldGroup}>
            <label style={styles.label}>Username</label>
            <input
              style={styles.input}
              type="text"
              value={form.username}
              onChange={handleChange("username")}
              disabled={isLoading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Full Name</label>
            <input
              style={styles.input}
              type="text"
              value={form.fullName}
              onChange={handleChange("fullName")}
              disabled={isLoading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Email</label>
            <input
              style={styles.input}
              type="email"
              value={form.email}
              onChange={handleChange("email")}
              autoComplete="email"
              disabled={isLoading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Phone Number</label>
            <input
              style={styles.input}
              type="tel"
              value={form.phone}
              onChange={handleChange("phone")}
              disabled={isLoading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Password</label>
            <input
              style={styles.input}
              type="password"
              value={form.password}
              onChange={handleChange("password")}
              autoComplete="new-password"
              disabled={isLoading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Confirm Password</label>
            <input
              style={styles.input}
              type="password"
              value={form.confirmPassword}
              onChange={handleChange("confirmPassword")}
              autoComplete="new-password"
              disabled={isLoading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Role</label>
            <select
              style={styles.select}
              value={form.role}
              onChange={handleChange("role")}
              disabled={isLoading}
            >
              <option value="">Select your role</option>
              {ROLES.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          {error && <p style={styles.error}>{error}</p>}

          <button type="submit" style={styles.submitButton} disabled={isLoading}>
            {isLoading ? "Creating account…" : "Sign up"}
          </button>
        </form>

        <p style={styles.switchText}>
          Already have an account?{" "}
          <button type="button" style={styles.switchLink} onClick={onGoLogin}>
            Login
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
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "14px",
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
  select: {
    padding: "8px 12px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "14px",
    outline: "none",
    width: "100%",
    boxSizing: "border-box",
    color: "#111827",
    backgroundColor: "#fff",
    cursor: "pointer",
    appearance: "auto",
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
