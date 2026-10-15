import { defineConfig, devices } from "@playwright/test";

// Browser smoke test of the built package: `npm run build && npm run test:browser`.
export default defineConfig({
  testDir: "browser",
  timeout: 300_000,
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: "http://localhost:4321" },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
  webServer: {
    command: "vite build --config browser/vite.config.ts && vite preview --config browser/vite.config.ts",
    url: "http://localhost:4321",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
