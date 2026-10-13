"""Upload only a model's card (tools/models/<model>/hub/README.md) to its Hugging Face repo.

Weight files are untouched, so cached downloads stay valid.
Usage: HF_TOKEN=... python upload_card.py <model> [repo_id]
"""

import json
import os
import sys

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: upload_card.py <model> [repo_id]")
  d = os.path.join(HERE, "models", sys.argv[1])
  with open(os.path.join(d, "model.json")) as f:
    repo_id = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else json.load(f)["repo"]
  api = HfApi(token=os.environ.get("HF_TOKEN"))
  if not api.repo_exists(repo_id, repo_type="model"):
    # A model whose weights were never released has no repo yet; its first
    # release-weights run uploads this card along with the weights.
    print(f"skipped: {repo_id} does not exist yet")
    return
  info = api.upload_file(
    path_or_fileobj=os.path.join(d, "hub", "README.md"),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message=os.environ.get("CARD_COMMIT_MESSAGE", "Update model card"),
  )
  print("uploaded:", info.commit_url)


if __name__ == "__main__":
  main()
