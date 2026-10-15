// Runs the packaged build the way an app would: bundled by Vite, model in a Web Worker on WASM.
import { clearCache, loadForecaster } from "forecast-web";

declare global {
  interface Window {
    smoke: Promise<unknown>;
  }
}

async function run() {
  const model = await loadForecaster({ model: "chronos-bolt-tiny", device: "wasm" });
  const series = Array.from({ length: 300 }, (_, i) => 10 + 3 * Math.sin((2 * Math.PI * i) / 24) + 0.3 * Math.sin(i * 1.7));
  const [forecast] = await model.forecast([series], { horizon: 24 });
  series[280] += 15;
  const check = await model.detectAnomalies(series);
  await model.dispose();
  const disposed = await model.forecast([series]).then(() => "resolved", (e: { code?: string }) => e.code);
  return {
    backend: model.backend,
    horizon: forecast.median.length,
    finite: [...forecast.median, ...forecast.q10, ...forecast.q90].every(Number.isFinite),
    ordered: forecast.q10.every((v, i) => v <= forecast.median[i] && forecast.median[i] <= forecast.q90[i]),
    flagged: check.anomalies.map((a) => a.index),
    disposed,
    cleared: await clearCache(),
  };
}

window.smoke = run();
window.smoke.then(
  (r) => (document.getElementById("out")!.textContent = JSON.stringify(r)),
  (e) => (document.getElementById("out")!.textContent = `error: ${e}`),
);
