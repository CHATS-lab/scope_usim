import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0d0f14",
        panel: "#161922",
        panelAlt: "#1c2030",
        border: "#262b3a",
        text: "#e5e7eb",
        muted: "#9ca3af",
        accent: "#7c9cff",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
