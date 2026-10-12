"""Upload build/hub to a Hugging Face model repo in a single commit.

Weight files on the Hub that are not part of this upload are deleted, so a
renamed or dropped format never lingers next to the new config.json.

Usage: HF_TOKEN=... uv run python upload_hub.py <repo_id> [commit message]
Meant to run from the release-weights GitHub Actions workflow.
"""

import os
import sys

from huggingface_hub import HfApi

HUB = os.path.join(os.path.dirname(__file__), "..", "build", "hub")
REQUIRED = ["config.json", "README.md", "onnx/model_quantized.onnx", "onnx/model_fp16w.onnx", "onnx/model_fp16w.f16"]


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: upload_hub.py <repo_id> [commit message]")
  repo_id = sys.argv[1]
  message = sys.argv[2] if len(sys.argv) > 2 else "Update weights"
  missing = [f for f in REQUIRED if not os.path.exists(os.path.join(HUB, f))]
  if missing:
    sys.exit(f"build/hub is incomplete, missing: {missing}")

  api = HfApi(token=os.environ.get("HF_TOKEN"))
  api.create_repo(repo_id, repo_type="model", exist_ok=True)
  info = api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=HUB,
    commit_message=message,
    delete_patterns=["*.onnx", "*.f16", "*.onnx_data"],
  )
  print("uploaded:", info.commit_url)
  print("revision:", info.oid)
  summary = os.environ.get("GITHUB_STEP_SUMMARY")
  if summary:
    with open(summary, "a") as f:
      f.write(f"Uploaded to [{repo_id}](https://huggingface.co/{repo_id}) at revision `{info.oid}`.\n")


if __name__ == "__main__":
  main()
