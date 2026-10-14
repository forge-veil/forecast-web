"""Export a TabICLv2 network (classifier or regressor) to a single-file ONNX graph.

The graph is the network only: column embedding, row interaction, in-context
learning and the decoder. Encoding, the ensemble, preprocessing and output
post-processing run in the JS caller (src/tabular), mirroring tabicl 2.2.0.

Inputs : X        [B, T, H] float32, preprocessed features, train rows first
         y_train  [B, N]    float32, class index or standardized target
Output : logits    [B, T-N, 10]  (classifier, raw logits)
         quantiles [B, T-N, 999] (regressor, raw quantiles)

After export, the graph is checked against PyTorch on shapes not used for the
trace, so a data-dependent branch baked in by tracing is caught.

Usage: uv run python models/tabicl/export_onnx.py <model> [out]
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..")


def load_meta(name):
  path = os.path.join(TOOLS, "models", name, "model.json")
  if not os.path.isfile(path):
    sys.exit(f"unknown model {name!r}: no tools/models/{name}/model.json")
  with open(path) as f:
    meta = json.load(f)
  if meta.get("scripts") != "tabicl":
    sys.exit(f"{name} is not a TabICL model")
  return meta


def patch_for_export():
  """Replace upstream's data-dependent skip branches with torch.where (same values).

  SkippableLinear and InducedSelfAttentionBlock assign skip values through
  boolean-mask indexing behind `if mask.any()`, which tracing freezes to the
  sample's shapes. Computing everything and selecting with torch.where is
  equivalent and exports as a shape-agnostic Where.
  """
  import torch.nn.functional as F
  from tabicl._model import layers

  def skippable_linear(self, src):
    out = F.linear(src, self.weight, self.bias)
    skip = (src == self.skip_value).all(dim=-1, keepdim=True)
    return torch.where(skip, torch.full_like(out, self.skip_value), out)

  def isab(self, src, train_size=None):
    out = self.induced_attention(src, train_size)
    skip = (src == self.skip_value).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
    return torch.where(skip, torch.full_like(out, self.skip_value), out)

  layers.SkippableLinear.forward = skippable_linear
  layers.InducedSelfAttentionBlock.forward = isab

  # QASSMax scales queries by an MLP of log(n), n = number of keys. Upstream takes the
  # log with math.log on a Python int, which tracing freezes; torch.log keeps n dynamic.
  from tabicl._model import ssmax

  def logn(n, device, dtype):
    return torch.log(torch.clamp(torch.as_tensor(n, device=device, dtype=dtype), min=1))

  ssmax._logn = logn

  # Classification slices logits to len(unique(y_train[0])), which tracing freezes to the
  # sample's class count. Export all 10 logits; the JS caller takes the first C.
  from collections import OrderedDict

  from tabicl._model import learning

  def predict_standard(self, R, y_train, return_logits=False, softmax_temperature=0.9, auto_batch=True):
    out = self.inference_mgr(self._icl_predictions, inputs=OrderedDict([("R", R), ("y_train", y_train)]), auto_batch=auto_batch)
    return out[:, y_train.shape[1] :]

  learning.ICLearning._predict_standard = predict_standard


def load_network(meta):
  """The upstream network, loaded the way the estimator loads it."""
  from tabicl import TabICLClassifier, TabICLRegressor

  ckpt = hf_hub_download(meta["source"], meta["checkpoint"], revision=meta["revision"])
  cls = TabICLClassifier if meta["task"] == "classification" else TabICLRegressor
  est = cls(checkpoint_version=meta["checkpoint"], model_path=ckpt, allow_auto_download=False, device="cpu", use_amp=False, use_fa3=False)
  rng = np.random.default_rng(0)
  X = pd.DataFrame(rng.normal(size=(12, 2)))
  y = np.arange(12) % 2 if meta["task"] == "classification" else rng.normal(size=12)
  est.fit(X, y)
  # The estimator's inference config pins the device to CPU; the default would pick MPS or CUDA.
  return est.model_.eval(), est.inference_config_


class Graph(torch.nn.Module):
  def __init__(self, model, config):
    super().__init__()
    self.model = model
    self.config = config

  def forward(self, X, y_train):
    # Upstream inference path without caches: raw logits (classifier) or raw quantiles (regressor).
    return self.model._inference_forward(X, y_train, return_logits=True, inference_config=self.config)


def sample(b, n, m, h, classification, seed):
  g = torch.Generator().manual_seed(seed)
  X = torch.randn(b, n + m, h, generator=g)
  if classification:
    # Vary the class count per shape so a traced class-count dependence is caught.
    y = (torch.arange(n).repeat(b, 1) % (2 + seed % 9)).float()
  else:
    y = torch.randn(b, n, generator=g)
  return X, y


def out_path(name):
  return sys.argv[2] if len(sys.argv) > 2 else os.path.join(TOOLS, "..", "build", name, "onnx", "model.onnx")


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: export_onnx.py <model> [out]")
  name = sys.argv[1]
  meta = load_meta(name)
  classification = meta["task"] == "classification"
  out_name = "logits" if classification else "quantiles"
  w = Graph(*load_network(meta)).eval()
  patch_for_export()
  X, y = sample(2, 40, 10, 6, classification, 0)
  out = out_path(name)
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  with torch.no_grad():
    torch.onnx.export(
      w,
      (X, y),
      out,
      input_names=["X", "y_train"],
      output_names=[out_name],
      dynamic_axes={"X": {0: "batch", 1: "rows", 2: "features"}, "y_train": {0: "batch", 1: "train_rows"}, out_name: {0: "batch", 1: "test_rows"}},
      opset_version=17,
      dynamo=False,
      external_data=False,
    )
  print("exported", os.path.abspath(out), os.path.getsize(out) // 2**20, "MB")

  sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
  worst = 0.0
  for b, n, m, h, seed in ((1, 25, 7, 3, 1), (3, 100, 30, 12, 2), (2, 8, 2, 1, 3), (4, 60, 15, 20, 4)):
    X, y = sample(b, n, m, h, classification, seed)
    with torch.no_grad():
      ref = w(X, y).numpy()
    got = sess.run(None, {"X": X.numpy(), "y_train": y.numpy()})[0]
    assert got.shape == ref.shape, (got.shape, ref.shape)
    d = float(np.abs(got - ref).max())
    worst = max(worst, d)
    print(f"graph check B={b} N={n} M={m} H={h}: max|d| = {d:.3e}")
  assert worst < 1e-4, worst


if __name__ == "__main__":
  main()
