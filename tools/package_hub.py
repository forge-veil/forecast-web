"""Assemble build/<model> for upload: config.json with file sizes and hashes, plus the model card.

The loader reads `files` to show accurate download progress from the first
byte and to key its cache by content hash, so re-exported weights never mix
with stale cached copies.

Usage: uv run python package_hub.py <model>   (after that model's export scripts)
"""

import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def model_dir(name):
  d = os.path.join(HERE, "models", name)
  if not os.path.isfile(os.path.join(d, "model.json")):
    sys.exit(f"unknown model {name!r}: no tools/models/{name}/model.json")
  return d


def load_model(name):
  with open(os.path.join(model_dir(name), "model.json")) as f:
    return json.load(f)


def build_dir(name):
  return os.path.join(HERE, "..", "build", name)


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
      h.update(chunk)
  return h.hexdigest()


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: package_hub.py <model>")
  name = sys.argv[1]
  meta, src, hub = load_model(name), model_dir(name), build_dir(name)
  with open(os.path.join(src, "hub", "config.json")) as f:
    config = json.load(f)
  config["source_revision"] = meta["revision"]
  config["files"] = {}
  for rel in meta["files"]:
    p = os.path.join(hub, rel)
    if not os.path.exists(p):
      print("missing, skipped:", rel)
      continue
    config["files"][rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    print(rel, config["files"][rel]["bytes"] // 2**20, "MB")
  with open(os.path.join(hub, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
  shutil.copy(os.path.join(src, "hub", "README.md"), os.path.join(hub, "README.md"))
  print("wrote", os.path.abspath(os.path.join(hub, "config.json")))


if __name__ == "__main__":
  main()
