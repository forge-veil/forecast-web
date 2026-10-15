import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: { alias: { src: fileURLToPath(new URL("./src", import.meta.url)) } },
  test: {
    globals: true,
    environment: "node",
    testTimeout: 120_000,
    hookTimeout: 300_000,
    // The demo site in site/ has its own Vitest config; browser/ is the Playwright suite.
    exclude: ["**/node_modules/**", "site/**", "browser/**"],
    environmentMatchGlobs: [["test/react.test.tsx", "jsdom"]],
  },
});
