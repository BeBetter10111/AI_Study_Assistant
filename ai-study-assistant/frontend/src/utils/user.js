// Display-name priority: full name > username > login identifier.
export const displayName = (profile, fallback = "") =>
  profile?.full_name?.trim() || profile?.username || fallback || "Tài khoản";

// Use the first letters of the first and last words ("Le Ho Chi Hai" -> "LH"); use up to 2 letters for a single word.
export const initials = (name) => {
  const words = (name || "").trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
};

const ROLE_LABEL = { Student: "Học viên", Staff: "Nhân viên", Admin: "Quản trị viên" };
export const roleLabel = (role) => ROLE_LABEL[role] || role || "";
