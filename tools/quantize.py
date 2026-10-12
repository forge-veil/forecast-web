"""Write the fp16 and 8-bit variants next to build/hub/onnx/model.onnx.

model_quantized.onnx: weight-only MatMulNBits, 8 bits, block 32, symmetric.
model_fp16.onnx:      fp16 weights with fp32 inputs and outputs.
File names follow the Transformers.js convention (dtype q8 -> model_quantized).
"""

import os

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
  DefaultWeightOnlyQuantConfig,
  MatMulNBitsQuantizer,
)
from onnxruntime.transformers.float16 import convert_float_to_float16

D = os.path.join(os.path.dirname(__file__), "..", "build", "hub", "onnx")
SRC = os.path.join(D, "model.onnx")


def size(p):
  return f"{os.path.getsize(p) // 2**20} MB"


def main():
  m = onnx.load(SRC)
  cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8)
  q = MatMulNBitsQuantizer(m, algo_config=cfg)
  q.process()
  out = os.path.join(D, "model_quantized.onnx")
  onnx.save_model(q.model.model, out)
  print("q8", size(out))
  del m, q

  m = onnx.load(SRC)
  out = os.path.join(D, "model_fp16.onnx")
  onnx.save_model(convert_float_to_float16(m, keep_io_types=True), out)
  print("fp16", size(out))


if __name__ == "__main__":
  main()
