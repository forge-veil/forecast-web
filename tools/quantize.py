"""Write the compressed variants next to build/hub/onnx/model.onnx.

model_quantized.onnx: weight-only MatMulNBits, 8 bits, block 32, symmetric.
                      Runs on WASM only: onnxruntime-web's WebGPU MatMulNBits
                      supports 2 and 4 bits. 4 bits was measured and rejected
                      (mean error 3-7% of series std, worst over 50%).
model_fp16.onnx:      fp16 weight storage with fp32 compute. Each large weight
                      is stored as fp16 and cast to fp32 in the graph; ORT folds
                      the casts at session creation. A full fp16 conversion
                      produces wrong results on WebGPU, where reductions and
                      attention scores accumulate in half precision.
File names follow the Transformers.js convention (dtype q8 -> model_quantized).
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import (
  DefaultWeightOnlyQuantConfig,
  MatMulNBitsQuantizer,
)

D = os.path.join(os.path.dirname(__file__), "..", "build", "hub", "onnx")
SRC = os.path.join(D, "model.onnx")
MIN_ELEMENTS = 1024  # small tensors (norm scales, biases) stay fp32


def size(p):
  return f"{os.path.getsize(p) // 2**20} MB"


def q8():
  m = onnx.load(SRC)
  cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8)
  q = MatMulNBitsQuantizer(m, algo_config=cfg)
  q.process()
  out = os.path.join(D, "model_quantized.onnx")
  onnx.save_model(q.model.model, out)
  print("q8", size(out))


F16W_GRAPH = "model_fp16w.onnx"
F16W_DATA = "model_fp16w.f16"
F16W_LOCATION = "model_fp16w.onnx_data"  # the name the graph uses for the fp32-expanded buffer
ALIGN = 64  # fp32-space alignment of each tensor


def fp16_weights():
  """fp32 graph whose large weights live in external data stored as fp16.

  The graph's external-data offsets refer to the fp32 layout. The data file
  holds the same tensors as fp16 at exactly half those offsets, so the loader
  expands it with one pass and hands the result to onnxruntime as external data.
  Runs at fp32 speed on WebGPU from a half-size download.
  """
  m = onnx.load(SRC)
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
  graph_path = os.path.join(D, F16W_GRAPH)
  with open(graph_path, "wb") as f:
    f.write(m.SerializeToString())
  with open(os.path.join(D, F16W_DATA), "wb") as f:
    f.write(data)
  print("fp16 weights", size(graph_path), "graph +", size(os.path.join(D, F16W_DATA)), f"data ({moved} tensors)")


if __name__ == "__main__":
  q8()
  fp16_weights()
