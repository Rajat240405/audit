import { create } from "zustand";

export type Theme = "light" | "dark";

const STORAGE_KEY = "incois-theme";

function readInitialTheme(): Theme {
  try {
    const t = localStorage.getItem(STORAGE_KEY);
    // An explicit stored preference wins; no preference (or a garbage value)
    // starts in LIGHT mode — the application default.
    return t === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

function apply(theme: Theme) {
  if (typeof document !== "undefined") {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }
}

interface ThemeState {
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
}

// apply immediately so the UI matches on first paint (the index.html inline
// script already does this pre-paint; this keeps state consistent).
const initial = readInitialTheme();
apply(initial);

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: initial,
  setTheme: (t) => {
    apply(t);
    try {
      localStorage.setItem(STORAGE_KEY, t);
    } catch {
      /* ignore */
    }
    set({ theme: t });
  },
  toggleTheme: () => get().setTheme(get().theme === "dark" ? "light" : "dark"),
}));
