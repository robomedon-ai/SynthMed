"""Restore the production checkpoints into the app tree at container startup.

Idempotent: any checkpoint already present is skipped, so restarts are cheap.
Reads the same manifest used for upload, so the two never drift.

Env vars (set as Space secrets/variables):
    HF_MODEL_REPO   required   e.g. user/synthmed-checkpoints
    HF_TOKEN        optional   only for a private checkpoint repo
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoints_manifest import PRODUCTION_CKPTS, existing, missing  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    repo = os.environ.get("HF_MODEL_REPO")
    if not repo:
        print("[checkpoints] HF_MODEL_REPO not set — skipping download "
              "(assuming checkpoints are baked into the image).", flush=True)
        return

    need = missing(ROOT)
    if not need:
        print(f"[checkpoints] all {len(PRODUCTION_CKPTS)} present — skipping download.",
              flush=True)
        return

    from huggingface_hub import snapshot_download
    print(f"[checkpoints] downloading {len(need)} files from {repo} ...", flush=True)
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=str(ROOT),
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=need,
    )

    still = missing(ROOT)
    if still:
        print("[checkpoints] ERROR — missing after download:", file=sys.stderr)
        for p in still:
            print("   -", p, file=sys.stderr)
        sys.exit(1)
    print(f"[checkpoints] ready ({len(existing(ROOT))}/{len(PRODUCTION_CKPTS)}).",
          flush=True)


if __name__ == "__main__":
    main()
