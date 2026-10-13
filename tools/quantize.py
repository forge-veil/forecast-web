"""Write the compressed variants next to build/<model>/onnx/model.onnx.

model_quantized.onnx: weight-only MatMulNBits, 8 bits, block 32, symmetric.
                      Runs on WASM only: onnxruntime-web's WebGPU MatMulNBits
                      supports 2 and 4 bits. 4 bits was measured and rejected
                      (mean error 3-7% of series std, worst over 50%).
model_fp16w.onnx +    fp32 graph whose large weights are external data, shipped
model_fp16w.f16       as fp16 and expanded to fp32 by the loader. A standard fp16
                      conversion gives wrong results on WebGPU (half-precision
                      accumulation), and in-graph Casts are not folded on WebGPU,
                      which made every forecast slow.
fp32 and q8 follow Transformers.js file names (dtype q8 -> model_quantized).
Only variants whose files are listed in tools/models/<model>/model.json are built.

Usage: uv run python quantize.py <model>
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import (
  DefaultWeightOnlyQuantConfig,
  MatMulNBitsQuantizer,
)

from package_hub import build_dir, load_model

MIN_ELEMENTS = 1024  # small tensors (norm scales, biases) stay fp32


def size(p):
  return f"{os.path.getsize(p) // 2**20} MB"


def q8(d):
  m = onnx.load(os.path.join(d, "model.onnx"))
  cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8)
  q = MatMulNBitsQuantizer(m, algo_config=cfg)
  q.process()
  out = os.path.join(d, "model_quantized.onnx")
  onnx.save_model(q.model.model, out)
  print("q8", size(out))


F16W_GRAPH = "model_fp16w.onnx"
F16W_DATA = "model_fp16w.f16"
F16W_LOCATION = "model_fp16w.onnx_data"  # the name the graph uses for the fp32-expanded buffer
ALIGN = 64  # fp32-space alignment of each tensor


def fp16_weights(d):
  """fp32 graph whose large weights live in external data stored as fp16.

  The graph's external-data offsets refer to the fp32 layout. The data file
  holds the same tensors as fp16 at exactly half those offsets, so the loader
  expands it with one pass and hands the result to onnxruntime as external data.
  Runs at fp32 speed on WebGPU from a half-size download.
  """
  m = onnx.load(os.path.join(d, "model.onnx"))
  data = bytearray()
  moved = 0
  for init in m.graph.initializer:
    n = int(np.prod(init.dims)) if init.dims else 1
    if init.data_type != TensorProto.FLOAT or n < MIN_ELEMENTS:
      continue
    half = numpy_helper.to_array(init).astype(np.float16).tobytes()
    pad = (-len(data)) % (ALIGN // 2)
    data.extend(b"\0" * pad)
    offset32 = 2 * len(data)
    data.extend(half)
    init.ClearField("raw_data")
    init.ClearField("float_data")
    init.data_location = TensorProto.EXTERNAL
    del init.external_data[:]
    for k, v in (("location", F16W_LOCATION), ("offset", str(offset32)), ("length", str(4 * n))):
      e = init.external_data.add()
      e.key, e.value = k, v
    moved += 1
  graph_path = os.path.join(d, F16W_GRAPH)
  with open(graph_path, "wb") as f:
    f.write(m.SerializeToString())
  with open(os.path.join(d, F16W_DATA), "wb") as f:
    f.write(data)
  print("fp16 weights", size(graph_path), "graph +", size(os.path.join(d, F16W_DATA)), f"data ({moved} tensors)")


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: quantize.py <model>")
  name = sys.argv[1]
  files = load_model(name)["files"]
  d = os.path.join(build_dir(name), "onnx")
  if "onnx/model_quantized.onnx" in files:
    q8(d)
  if "onnx/model_fp16w.onnx" in files:
    fp16_weights(d)


if __name__ == "__main__":
  main()
