"""Generate parity fixtures for a Chronos-Bolt size from the official pipeline.

Writes test/fixtures/<model>/reference.json: each input series and the official
ChronosBoltPipeline.predict_quantiles() output (q10..q90) at horizons 64 and
200, plus a graph check of the fp32 ONNX graph (with JS-style normalization and
left-padding) against the official forward pass on unpadded input.

Usage: uv run python models/chronos-bolt/make_reference.py <model>
"""

import json
import math
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

from export_onnx import C, TOOLS, load_meta, load_pipeline

HORIZONS = [64, 200]
Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def series():
  rng = np.random.default_rng(7)
  t = np.arange(600)
  s = {}
  s["sine"] = (10 * np.sin(2 * np.pi * t / 24)).tolist()
  s["trend_noise"] = (0.5 * t + rng.normal(0, 5, 600) + 200).tolist()
  s["short_50"] = (np.sin(np.arange(50) / 3) * 4 + 20).tolist()
  s["short_7"] = [3.0, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5]
  inter = (np.cos(2 * np.pi * np.arange(300) / 30) * 3 + 50).tolist()
  for i in (40, 41, 42, 150, 151, 299):
    inter[i] = None
  s["interior_nans"] = inter
  s["leading_nans"] = [None] * 20 + (np.arange(180) * 0.1 + rng.normal(0, 0.3, 180)).tolist()
  s["positive_counts"] = rng.poisson(lam=3 + 2 * np.sin(np.arange(400) / 10)).astype(float).tolist()
  s["constant"] = [5.0] * 100
  s["long_3000"] = (np.sin(np.arange(3000) / 50) * 100 + rng.normal(0, 2, 3000)).tolist()
  s["negative"] = (-1000 - 3 * np.arange(256) + rng.normal(0, 10, 256)).tolist()
  s["tiny_scale"] = (1e-4 * np.sin(np.arange(200) / 5)).tolist()
  s["exact_2048"] = (np.cos(np.arange(2048) / 7) * 20 + np.arange(2048) * 0.05 + rng.normal(0, 1, 2048)).tolist()
  return s


def to_np(v):
  return np.array([np.nan if x is None else x for x in v], dtype=np.float64)


def normalize(x):
  """Mirror of the JS preprocessing: last 2048 points, instance norm, left-pad, mask."""
  x = x[-C:]
  obs = ~np.isnan(x)
  loc = x[obs].mean() if obs.any() else 0.0
  scale = math.sqrt(((x[obs] - loc) ** 2).mean()) if obs.any() else 1.0
  scale = scale if scale != 0 else 1e-5
  ctx = np.zeros(C, np.float32)
  mk = np.zeros(C, np.float32)
  ctx[C - len(x) :] = np.where(obs, (x - loc) / scale, 0.0)
  mk[C - len(x) :] = obs
  return ctx, mk, loc, scale


def graph_check(pipe, sess, s):
  worst = 0.0
  for name in ("short_7", "short_50", "interior_nans", "long_3000"):
    x = to_np(s[name])
    ctx, mk, loc, scale = normalize(x)
    q = sess.run(None, {"context": ctx[None], "mask": mk[None]})[0][0] * scale + loc
    with torch.no_grad():
      ref = pipe.model(context=torch.tensor(x[-C:], dtype=torch.float32)[None]).quantile_preds[0].numpy()
    sd = float(np.nanstd(x[-C:])) or 1.0
    worst = max(worst, float(np.abs(q - ref).max()) / sd)
  return worst


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: make_reference.py <model>")
  name = sys.argv[1]
  meta = load_meta(name)
  pipe = load_pipeline(meta)
  s = series()
  names = list(s)
  onnx_path = os.path.join(TOOLS, "..", "build", name, "onnx", "model.onnx")
  sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
  g = graph_check(pipe, sess, s)
  print(f"graph check fp32 vs official forward: max|d|/std = {g:.3e}")
  assert g < 1e-3, g

  outputs = {}
  for h in HORIZONS:
    ctx = [torch.tensor(to_np(s[n]), dtype=torch.float32) for n in names]
    q, _ = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=Q)
    outputs[h] = q.numpy()  # [B, h, 9]

  cases = []
  for i, n in enumerate(names):
    for h in HORIZONS:
      assert np.isfinite(outputs[h][i]).all(), (n, h)
    cases.append(
      {
        "name": n,
        "input": [None if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in s[n]],
        "outputs": {str(h): outputs[h][i].astype(float).tolist() for h in HORIZONS},
      }
    )
  out = os.path.join(TOOLS, "..", "test", "fixtures", name, "reference.json")
  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, "w") as f:
    json.dump(
      {
        "source": f"chronos-forecasting 2.3.2 ChronosBoltPipeline.predict_quantiles {meta['source']}@{meta['revision'][:7]}",
        "horizons": HORIZONS,
        "graph_check": g,
        "cases": cases,
      },
      f,
    )
  print("wrote", os.path.abspath(out), len(cases), "cases")


if __name__ == "__main__":
  main()
