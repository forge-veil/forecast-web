import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: { alias: { src: fileURLToPath(new URL("./src", import.meta.url)) } },
  test: {
    globals: true,
    environment: "node",
    testTimeout: 120_000,
    environmentMatchGlobs: [["test/react.test.tsx", "jsdom"]],
  },
});
