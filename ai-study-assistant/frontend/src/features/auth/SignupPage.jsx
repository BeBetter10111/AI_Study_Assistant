import { useState } from "react";
import { api, registerUser } from "../../services/api";
import { authStyles as styles } from "./authStyles";

const extractErrorMessage = (err) => {
  if (!err.response) {
    // Không có response = request không tới được backend hoặc bị CORS chặn.
    // Hiện rõ URL đã gọi và origin của trang để biết ngay chỗ sai.
    const target = api.defaults.baseURL || "(VITE_API_BASE_URL chưa được cấu hình)";
    return (
      `Không thể kết nối tới server (Network Error) tại ${target}. ` +
      `Vui lòng kiểm tra backend đang chạy và ALLOWED_ORIGINS cho phép ${window.location.origin}.`
    );
  }
  const detail = err.response.data?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return err.message || "Sign up failed. Please try again.";
};

export default function SignupPage({ onGoLogin }) {
  const [form, setForm] = useState({
    username: "",
    fullName: "",
    email: "",
    phone: "",
    password: "",
    confirmPassword: "",
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
      await registerUser({
        username: form.username.trim(),
        full_name: form.fullName.trim() || null,
        email: form.email.trim(),
        phone: form.phone.trim() || null,
        password: form.password,
        role: "Staff",
      });
      onGoLogin("Sign up succeed! Please login.", form.email.trim());
    } catch (err) {
      setError(extractErrorMessage(err));
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

          {error && <p style={styles.error}>{error}</p>}

          <button type="submit" style={styles.submitButton} disabled={isLoading}>
            {isLoading ? "Creating account…" : "Sign up"}
          </button>
        </form>

        <p style={styles.switchText}>
          Already have an account?{" "}
          <button type="button" style={styles.switchLink} onClick={() => onGoLogin()}>
            Login
          </button>
        </p>
      </div>
    </div>
  );
}
