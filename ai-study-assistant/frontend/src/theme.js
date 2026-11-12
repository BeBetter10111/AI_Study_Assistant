const KEY = "theme";
const query = () => window.matchMedia("(prefers-color-scheme: dark)");

export const getThemePreference = () => {
  try {
    const saved = localStorage.getItem(KEY);
    return saved === "light" || saved === "dark" ? saved : "system";
  } catch {
    return "system";
  }
};

const resolve = (pref) => (pref === "system" ? (query().matches ? "dark" : "light") : pref);

export const applyTheme = (pref) => {
  document.documentElement.dataset.theme = resolve(pref);
};

export const setThemePreference = (pref) => {
  try {
    if (pref === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, pref);
  } catch {
    /* localStorage không dùng được: vẫn áp dụng cho phiên hiện tại */
  }
  applyTheme(pref);
};

// Gọi một lần khi khởi động: theo dõi thay đổi giao diện của hệ điều hành khi chọn "system".
export const initTheme = () => {
  applyTheme(getThemePreference());
  query().addEventListener("change", () => {
    if (getThemePreference() === "system") applyTheme("system");
  });
};
