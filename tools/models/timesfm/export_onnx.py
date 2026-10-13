"""Export TimesFM 2.5 200M to a single-file ONNX graph.

The graph is one prefill pass of TimesFM_2p5_200M_torch_module.decode() for a
fixed context, which covers horizons up to 128 with no autoregressive steps:
running per-patch stats, RevIN, transformer, un-RevIN of the last patch.

Inputs : inputs [B, C] float32, masks [B, C] float32 (1 = padded)
Outputs: point   [B, 128, 10] channel 0 = mean head, 1..9 = q10..q90
         qspread [B, 128, 10] first 128 steps of the continuous quantile head

Forecast-level processing (input normalization, flip invariance, continuous
quantile head, quantile crossing fix, positivity) runs in the JS caller.

Usage: uv run python models/timesfm/export_onnx.py timesfm [context=512] [out=../build/timesfm/onnx/model.onnx]
Export approach adapted from hwkim3330/timesfm-web (Apache-2.0).
"""

import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
from timesfm.torch import transformer, util

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "model.json")) as f:
  _MODEL = json.load(f)
REPO = _MODEL["source"]
# Pinned in model.json so a rebuild in CI always starts from the same checkpoint.
REVISION = _MODEL["revision"]
# argv: <model> [context=512] [out]
if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "timesfm"):
  sys.exit("usage: export_onnx.py timesfm [context=512] [out]")
C = int(sys.argv[2]) if len(sys.argv) > 2 else 512
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "..", "..", "build", "timesfm", "onnx", "model.onnx")
HORIZON = 128


def stack_narrow(ts):
  """torch.stack(ts, 1) built from 2-input concats.

  A single 16-input Concat exceeds WebGPU's per-shader storage buffer limit
  (8 by default) and produces wrong results on onnxruntime-web's WebGPU EP.
  """
  ts = [t.unsqueeze(1) for t in ts]
  while len(ts) > 1:
    ts = [torch.cat(ts[i : i + 2], 1) if i + 1 < len(ts) else ts[i] for i in range(0, len(ts), 2)]
  return ts[0]


class Prefill(torch.nn.Module):
  def __init__(self, m: TimesFM_2p5_200M_torch_module):
    super().__init__()
    self.m = m

  def forward(self, inputs, masks):
    m = self.m
    b = inputs.shape[0]
    p = C // m.p
    x = inputs.reshape(b, p, m.p)
    mk = masks.reshape(b, p, m.p) > 0.5
    n = torch.zeros_like(inputs[:, 0])
    mu = torch.zeros_like(n)
    sg = torch.zeros_like(n)
    mus, sgs = [], []
    for i in range(p):
      (n, mu, sg), _ = util.update_running_stats(n, mu, sg, x[:, i], mk[:, i])
      mus.append(mu)
      sgs.append(sg)
    cmu = stack_narrow(mus)
    csg = stack_narrow(sgs)
    xn = torch.where(mk, 0.0, util.revin(x, cmu, csg, reverse=False))
    h = m.tokenizer(torch.cat([xn, mk.to(xn.dtype)], -1))
    pad = mk[..., -1]
    for layer in m.stacked_xf:
      h, _ = layer(h, pad, None)
    last = h[:, -1]
    lmu = cmu[:, -1:]
    lsg = csg[:, -1:]
    point = m.output_projection_point(last) * lsg + lmu
    qs = m.output_projection_quantiles(last) * lsg + lmu
    return point.reshape(b, m.o, m.q), qs.reshape(b, m.os, m.q)[:, :HORIZON]


def main():
  m = TimesFM_2p5_200M_torch_module()
  m.load_checkpoint(hf_hub_download(REPO, "model.safetensors", revision=REVISION))
  m.to("cpu").eval()
  for layer in m.stacked_xf:
    # Explicit masked softmax: same math as the fused SDPA path, export friendly.
    layer.attn.attention_fn = transformer._dot_product_attention
  w = Prefill(m).eval()

  x = torch.randn(2, C).cumsum(-1) + 100
  mk = torch.zeros(2, C)
  mk[1, :100] = 1
  os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
  with torch.no_grad():
    torch.onnx.export(
      w,
      (x, mk),
      OUT,
      input_names=["inputs", "masks"],
      output_names=["point", "qspread"],
      dynamic_axes={k: {0: "batch"} for k in ("inputs", "masks", "point", "qspread")},
      opset_version=17,
      dynamo=False,
      external_data=False,
    )
  print("exported", os.path.abspath(OUT), os.path.getsize(OUT) // 2**20, "MB")


if __name__ == "__main__":
  main()
