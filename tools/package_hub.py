"""Assemble build/hub for upload: config.json with file sizes and hashes, plus the model card.

The loader reads `files` to show accurate download progress from the first
byte and to key its cache by content hash, so re-exported weights never mix
with stale cached copies.

Usage: uv run python package_hub.py   (after export_onnx.py and quantize.py)
"""

import hashlib
import json
import os
import shutil

from export_onnx import REVISION

HERE = os.path.dirname(__file__)
HUB = os.path.join(HERE, "..", "build", "hub")
FILES = [
  "onnx/model.onnx",
  "onnx/model_fp16w.onnx",
  "onnx/model_fp16w.f16",
  "onnx/model_quantized.onnx",
]


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
      h.update(chunk)
  return h.hexdigest()


def main():
  with open(os.path.join(HERE, "hub", "config.json")) as f:
    config = json.load(f)
  config["source_revision"] = REVISION
  config["files"] = {}
  for rel in FILES:
    p = os.path.join(HUB, rel)
    if not os.path.exists(p):
      print("missing, skipped:", rel)
      continue
    config["files"][rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    print(rel, config["files"][rel]["bytes"] // 2**20, "MB")
  with open(os.path.join(HUB, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
  shutil.copy(os.path.join(HERE, "hub", "README.md"), os.path.join(HUB, "README.md"))
  print("wrote", os.path.abspath(os.path.join(HUB, "config.json")))


if __name__ == "__main__":
  main()
