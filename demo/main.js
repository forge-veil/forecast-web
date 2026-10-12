import { loadTimesFM } from "../dist/index.js";

const q = new URLSearchParams(location.search);
const statusEl = document.getElementById("status");
const say = (t) => (statusEl.textContent = t);

// Hourly-ish signal: daily cycle, slow trend, noise, and a few missing points.
const history = Array.from({ length: 400 }, (_, i) => {
  const v = 50 + 0.05 * i + 10 * Math.sin((2 * Math.PI * i) / 24) + 2 * Math.sin(i * 1.7);
  return i % 97 === 50 ? null : v;
});

/** Runs the Python reference fixtures in the browser and reports error as a fraction of each series' std. */
async function parity(model) {
  const ref = (await import("../test/fixtures/reference.json")).default;
  const out = await model.forecast(ref.cases.map((c) => c.input), { horizon: ref.horizon });
  const keys = ["mean", "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"];
  return ref.cases.map((c, i) => {
    const v = c.input.filter((x) => x !== null).slice(-512);
    const mu = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length) || 1;
    let worst = 0, sum = 0;
    c.output.forEach((row, t) => keys.forEach((k, q) => {
      const d = Math.abs(out[i][k][t] - row[q]);
      worst = Math.max(worst, d);
      sum += d;
    }));
    return { name: c.name, worst: worst / sd, mean: sum / (c.output.length * keys.length) / sd };
  });
}

async function main() {
  const t0 = performance.now();
  const model = await loadTimesFM({
    model: q.get("model") ?? "/",
    dtype: q.get("dtype") ?? "auto",
    device: q.get("device") ?? "auto",
    worker: q.get("worker") !== "0",
    onProgress: ({ loaded, total, cached }) =>
      say(`Downloading ${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB${cached ? " (cached)" : ""}`),
  });
  const loadMs = performance.now() - t0;

  say(`Running on ${model.backend}…`);
  await model.forecast([history], { horizon: 48 }); // warm-up (shader compile on WebGPU)
  const times = [];
  let f;
  for (let i = 0; i < 5; i++) {
    const t1 = performance.now();
    [f] = await model.forecast([history], { horizon: 48 });
    times.push(performance.now() - t1);
  }
  const runMs = times.sort((x, y) => x - y)[2]; // median of 5

  say(`${model.backend} · ${model.dtype} · load ${(loadMs / 1000).toFixed(1)} s · forecast ${runMs.toFixed(0)} ms`);
  draw(history.slice(-120), f);
  const parityRows = q.get("parity") ? await parity(model) : undefined;
  window.__result = { backend: model.backend, dtype: model.dtype, loadMs, runMs, median: f.median, q10: f.q10, q90: f.q90, parity: parityRows };
  await model.dispose();
}

function draw(hist, f) {
  const W = 760, H = 300, P = 24;
  const all = [...hist.filter((v) => v != null), ...f.q10, ...f.q90];
  const lo = Math.min(...all), hi = Math.max(...all);
  const n = hist.length + f.horizon;
  const x = (i) => P + (i / (n - 1)) * (W - 2 * P);
  const y = (v) => H - P - ((v - lo) / (hi - lo)) * (H - 2 * P);
  const path = (pts) => pts.map(([i, v], k) => `${k ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const off = hist.length;
  const band =
    path(f.q90.map((v, i) => [off + i, v])) +
    f.q10.map((v, i) => [off + i, v]).reverse().map(([i, v]) => `L${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("") + "Z";
  const histPts = hist.map((v, i) => [i, v]).filter(([, v]) => v != null);
  document.getElementById("chart").innerHTML = `
    <path d="${band}" fill="var(--band)" />
    <path d="${path(histPts)}" fill="none" stroke="var(--muted)" stroke-width="1.5" />
    <path d="${path(f.median.map((v, i) => [off + i, v]))}" fill="none" stroke="var(--line)" stroke-width="2" />
    <line x1="${x(off)}" x2="${x(off)}" y1="${P}" y2="${H - P}" stroke="var(--muted)" stroke-dasharray="3 4" />`;
}

main().catch((e) => {
  say(`Error: ${e.code ?? ""} ${e.message}`);
  window.__error = { code: e.code, message: e.message };
});
