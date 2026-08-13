import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// Vite dev server proxies /api to the FastAPI backend so browser code always
// calls same-origin relative URLs (works in sandbox preview and on user machine).
const BACKEND_PORT = process.env.BACKEND_PORT || "8000";
const BACKEND_HOST = process.env.BACKEND_HOST || "127.0.0.1";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    // allow the sandbox preview host and any host (dev convenience)
    allowedHosts: true,
    proxy: {
      "/api": {
        target: `http://${BACKEND_HOST}:${BACKEND_PORT}`,
        changeOrigin: true,
        // SSE streaming needs proxy buffering disabled
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            proxyReq.setHeader("connection", "keep-alive");
          });
        },
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
