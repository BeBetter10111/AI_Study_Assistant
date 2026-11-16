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
    /* If localStorage is unavailable, still apply the theme for the current session. */
  }
  applyTheme(pref);
};

// Call once at startup: watch for operating-system theme changes when "system" is selected.
export const initTheme = () => {
  applyTheme(getThemePreference());
  query().addEventListener("change", () => {
    if (getThemePreference() === "system") applyTheme("system");
  });
};
