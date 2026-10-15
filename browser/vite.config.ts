import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Resolves "forecast-web" to the built package, so the test covers what npm ships.
export default defineConfig({
  root: fileURLToPath(new URL("./app", import.meta.url)),
  resolve: { alias: { "forecast-web": fileURLToPath(new URL("../dist/index.js", import.meta.url)) } },
  build: { outDir: fileURLToPath(new URL("./.out", import.meta.url)), emptyOutDir: true },
  preview: { port: 4321, strictPort: true },
});
