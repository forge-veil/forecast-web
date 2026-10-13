"""Export a Chronos-Bolt checkpoint to a single-file ONNX graph.

The graph covers patching, the residual patch embedding, the T5 encoder with
the [REG] token, the one-step decoder and the quantile head, for a fixed
2048-point context. Instance normalization, NaN handling, de-normalization and
the long-horizon rollout run in the JS caller, mirroring chronos_bolt.py.

Inputs : context   [B, 2048] float32, normalized, 0 where missing or padded (left-padded)
         mask      [B, 2048] float32, 1 = observed
Output : quantiles [B, 9, 64] float32, normalized q10..q90

Usage: uv run python models/chronos-bolt/export_onnx.py <model> [out]
"""

import json
import os
import sys

import torch
from chronos import ChronosBoltPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..")
C = 2048


def load_meta(name):
  path = os.path.join(TOOLS, "models", name, "model.json")
  if not os.path.isfile(path):
    sys.exit(f"unknown model {name!r}: no tools/models/{name}/model.json")
  with open(path) as f:
    meta = json.load(f)
  if meta.get("scripts") != "chronos-bolt":
    sys.exit(f"{name} is not a Chronos-Bolt model")
  return meta


def load_pipeline(meta):
  return ChronosBoltPipeline.from_pretrained(meta["source"], revision=meta["revision"])


class Graph(torch.nn.Module):
  def __init__(self, m):
    super().__init__()
    self.m = m
    cc = m.chronos_config
    assert cc.context_length == C and cc.use_reg_token, cc
    self.patch = cc.input_patch_size
    self.horizon = cc.prediction_length
    self.nq = m.num_quantiles

  def forward(self, context, mask):
    m = self.m
    b = context.shape[0]
    p = C // self.patch
    pc = context.reshape(b, p, self.patch)
    pm = mask.reshape(b, p, self.patch)
    pc = torch.where(pm > 0, pc, torch.zeros_like(pc))
    attn = (pm.sum(-1) > 0).to(context.dtype)
    emb = m.input_patch_embedding(torch.cat([pc, pm], -1))
    reg = m.shared.weight[m.config.reg_token_id].reshape(1, 1, -1).expand(b, 1, -1)
    emb = torch.cat([emb, reg], 1)
    attn = torch.cat([attn, torch.ones_like(attn[:, :1])], -1)
    hidden = m.encoder(attention_mask=attn, inputs_embeds=emb)[0]
    # Same as m.decode(), but with the start token passed as its embedding: the legacy
    # exporter types torch.full(..., decoder_start_token_id) as float, which breaks the Gather.
    start = m.shared.weight[m.config.decoder_start_token_id].reshape(1, 1, -1).expand(b, 1, -1)
    out = m.decoder(
      inputs_embeds=start,
      encoder_hidden_states=hidden,
      encoder_attention_mask=attn,
      use_cache=False,
      return_dict=True,
    ).last_hidden_state
    return m.output_patch_embedding(out).reshape(b, self.nq, self.horizon)


def out_path(name):
  return sys.argv[2] if len(sys.argv) > 2 else os.path.join(TOOLS, "..", "build", name, "onnx", "model.onnx")


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: export_onnx.py <model> [out]")
  name = sys.argv[1]
  pipe = load_pipeline(load_meta(name))
  w = Graph(pipe.model.eval()).eval()
  x = torch.randn(2, C)
  mk = torch.ones(2, C)
  mk[1, :1500] = 0
  x = x * mk
  out = out_path(name)
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  with torch.no_grad():
    torch.onnx.export(
      w,
      (x, mk),
      out,
      input_names=["context", "mask"],
      output_names=["quantiles"],
      dynamic_axes={k: {0: "batch"} for k in ("context", "mask", "quantiles")},
      opset_version=17,
      dynamo=False,
      external_data=False,
    )
  print("exported", os.path.abspath(out), os.path.getsize(out) // 2**20, "MB")


if __name__ == "__main__":
  main()
