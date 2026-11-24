import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The browser only talks to the Vite dev server; /api is forwarded to FastAPI.
// The Gemini API key lives only in the backend .env and never reaches this code.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
