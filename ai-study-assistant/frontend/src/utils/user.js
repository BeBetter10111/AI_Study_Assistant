// Tên hiển thị ưu tiên: họ tên > username > định danh lúc đăng nhập.
export const displayName = (profile, fallback = "") =>
  profile?.full_name?.trim() || profile?.username || fallback || "Tài khoản";

// Chữ cái đầu của từ đầu và từ cuối ("Lê Hồ Chí Hải" -> "LH"); một từ thì lấy tối đa 2 chữ cái.
export const initials = (name) => {
  const words = (name || "").trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
};

const ROLE_LABEL = { Student: "Học viên", Staff: "Nhân viên", Admin: "Quản trị viên" };
export const roleLabel = (role) => ROLE_LABEL[role] || role || "";
