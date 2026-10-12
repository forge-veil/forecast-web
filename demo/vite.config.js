import { defineConfig } from "vite";

// Serves the locally exported weights (build/hub) at the site root, standing in for the Hugging Face repo.
export default defineConfig({
  root: __dirname,
  publicDir: "../build/hub",
  server: { port: 5199, strictPort: true, fs: { allow: [".."] } },
});
