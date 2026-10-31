import { authStyles as styles } from "./authStyles";

// "Login or Sign up" ở cuối mỗi bước của luồng quên mật khẩu.
export default function AuthFooterLinks({ onGoLogin, onGoSignup }) {
  return (
    <p style={styles.switchText}>
      <button type="button" style={styles.switchLink} onClick={onGoLogin}>
        Login
      </button>{" "}
      or{" "}
      <button type="button" style={styles.switchLink} onClick={onGoSignup}>
        Sign up
      </button>
    </p>
  );
}
