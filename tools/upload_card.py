"""Upload only the model card (tools/hub/README.md) to the Hugging Face repo.

Weight files are untouched, so cached downloads stay valid.
Usage: HF_TOKEN=... python upload_card.py <repo_id>
"""

import os
import sys

from huggingface_hub import HfApi

CARD = os.path.join(os.path.dirname(__file__), "hub", "README.md")


def main():
  if len(sys.argv) < 2:
    sys.exit("usage: upload_card.py <repo_id>")
  info = HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
    path_or_fileobj=CARD,
    path_in_repo="README.md",
    repo_id=sys.argv[1],
    repo_type="model",
    commit_message=os.environ.get("CARD_COMMIT_MESSAGE", "Update model card"),
  )
  print("uploaded:", info.commit_url)


if __name__ == "__main__":
  main()
