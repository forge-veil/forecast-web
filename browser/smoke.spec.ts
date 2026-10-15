import { expect, test } from "@playwright/test";

test("loads a model, forecasts, detects anomalies and cleans up", async ({ page }) => {
  page.on("pageerror", (e) => console.log("page error:", e.message));
  await page.goto("/");
  const result = await page.evaluate(() => window.smoke);
  expect(result).toMatchObject({ backend: "wasm", horizon: 24, finite: true, ordered: true, disposed: "DISPOSED", cleared: true });
  expect((result as { flagged: number[] }).flagged).toContain(280);
});
