"""Generate parity fixtures from the official TimesFM 2.5 forecast.

Writes test/fixtures/timesfm/reference.json with each input series and the official
output for all 10 channels at horizon 128, plus a graph check that compares
the exported fp32 ONNX graph against the torch prefill on the same inputs.

Usage: uv run python models/timesfm/make_reference.py timesfm
"""

import json
import math
import os

import numpy as np
import onnxruntime as ort
import timesfm
import torch

from export_onnx import REPO, REVISION

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
ONNX = os.path.join(ROOT, "build", "timesfm", "onnx", "model.onnx")
OUT = os.path.join(ROOT, "test", "fixtures", "timesfm", "reference.json")
H = 128


def series():
  rng = np.random.default_rng(7)
  t = np.arange(600)
  s = {}
  s["sine"] = (10 * np.sin(2 * np.pi * t / 24)).tolist()
  s["trend_noise"] = (0.5 * t + rng.normal(0, 5, 600) + 200).tolist()
  s["short_50"] = (np.sin(np.arange(50) / 3) * 4 + 20).tolist()
  inter = (np.cos(2 * np.pi * np.arange(300) / 30) * 3 + 50).tolist()
  for i in (40, 41, 42, 150, 151, 299):
    inter[i] = None
  s["interior_nans"] = inter
  lead = [None] * 20 + (np.arange(180) * 0.1 + rng.normal(0, 0.3, 180)).tolist()
  s["leading_nans"] = lead
  s["positive_counts"] = rng.poisson(lam=3 + 2 * np.sin(np.arange(400) / 10)).astype(float).tolist()
  s["constant"] = [5.0] * 100
  s["long_2000"] = (np.sin(np.arange(2000) / 50) * 100 + rng.normal(0, 2, 2000)).tolist()
  s["negative"] = (-1000 - 3 * np.arange(256) + rng.normal(0, 10, 256)).tolist()
  s["tiny_scale"] = (1e-4 * np.sin(np.arange(200) / 5)).tolist()
  s["exact_512"] = (np.cos(np.arange(512) / 7) * 20 + np.arange(512) * 0.05 + rng.normal(0, 1, 512)).tolist()
  return s


def to_np(v):
  return np.array([np.nan if x is None else x for x in v], dtype=np.float64)


def official(inputs):
  m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO, revision=REVISION, torch_compile=False)
  m.compile(
    timesfm.ForecastConfig(
      max_context=512,
      max_horizon=128,
      normalize_inputs=True,
      use_continuous_quantile_head=True,
      force_flip_invariance=True,
      infer_is_positive=True,
      fix_quantile_crossing=True,
    )
  )
  _, full = m.forecast(horizon=H, inputs=[to_np(v) for v in inputs])
  return full, m.model


def graph_check(module, sess):
  """fp32 ONNX graph vs torch decode() prefill on raw (unnormalized) inputs."""
  rng = np.random.default_rng(1)
  x = (rng.normal(0, 1, (3, 512)).cumsum(-1) + 50).astype(np.float32)
  mk = np.zeros((3, 512), dtype=np.float32)
  mk[1, :300] = 1
  mk[2, :480] = 1
  x[mk > 0] = 0
  pf, qs, _ = module.decode(H, torch.from_numpy(x), torch.from_numpy(mk > 0))
  ref_pt = pf[:, -1].numpy()
  ref_qs = qs[:, :H].numpy()
  pt, q = sess.run(None, {"inputs": x, "masks": mk})
  return float(np.abs(pt - ref_pt).max()), float(np.abs(q - ref_qs).max())


def main():
  s = series()
  names = list(s)
  full, module = official([s[n] for n in names])
  sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
  with torch.no_grad():
    gp, gq = graph_check(module, sess)
  print(f"graph check fp32 vs torch: point max|d|={gp:.3e} qspread max|d|={gq:.3e}")

  cases = []
  for i, n in enumerate(names):
    out = full[i]
    assert np.isfinite(out).all(), n
    cases.append(
      {
        "name": n,
        "input": [None if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in s[n]],
        "output": out.astype(float).tolist(),
      }
    )
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  with open(OUT, "w") as f:
    json.dump(
      {
        "source": "timesfm e51928e TimesFM_2p5_200M_torch.forecast",
        "horizon": H,
        "graph_check": {"point": gp, "qspread": gq},
        "cases": cases,
      },
      f,
    )
  print("wrote", os.path.abspath(OUT), len(cases), "cases")


if __name__ == "__main__":
  main()
