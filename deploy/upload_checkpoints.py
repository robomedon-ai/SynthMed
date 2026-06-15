"""Upload ONLY the production checkpoints to the Hugging Face Model Hub.

Run this ONCE locally, after training, to publish the ~6 GB production set. Each
file is uploaded at its app-relative path so the Space can restore the tree with
a single snapshot_download (see download_checkpoints.py).

    pip install huggingface_hub
    huggingface-cli login                       # once, needs a write token
    python deploy/upload_checkpoints.py --repo <user>/synthmed-checkpoints

Large files are stored via LFS automatically by the Hub.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoints_manifest import PRODUCTION_CKPTS, existing, missing  # noqa: E402

from huggingface_hub import HfApi, create_repo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="HF model repo id, e.g. user/synthmed-checkpoints")
    ap.add_argument("--private", action="store_true", help="create a private repo")
    args = ap.parse_args()

    files = existing(ROOT)
    absent = missing(ROOT)
    if absent:
        print(f"WARNING: {len(absent)} manifest checkpoints are missing locally "
              f"and will be skipped:")
        for p in absent:
            print("   -", p)
    if not files:
        sys.exit("No production checkpoints found locally. Nothing to upload.")

    total_gb = sum((ROOT / f).stat().st_size for f in files) / 1e9
    print(f"\nUploading {len(files)} checkpoints (~{total_gb:.1f} GB) to "
          f"{args.repo} ...\n")

    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f}")
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f,
                        repo_id=args.repo, repo_type="model")
    print("\nDone. Set HF_MODEL_REPO=%s in the Space and it will pull these at "
          "startup." % args.repo)


if __name__ == "__main__":
    main()
