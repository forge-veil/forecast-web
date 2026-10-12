import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/forecast-web/",
  plugins: [react()],
  resolve: { alias: { src: fileURLToPath(new URL("./src", import.meta.url)) } },
  server: { port: 5299, strictPort: true },
  preview: { port: 5299, strictPort: true },
  test: { globals: true, environment: "node" },
});
