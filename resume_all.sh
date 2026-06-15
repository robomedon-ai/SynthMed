#!/usr/bin/env bash
# Run this after a reboot to resume the TSTR work exactly where it stopped.
# Safe to run anytime: completed runs (best.pt present) are skipped.
cd /home/marija/Desktop/ferit/ROBOMED/godina1/wsi_app

# GPU driver can vanish after a kernel update — verify before launching, or
# PyTorch silently falls back to CPU and "training" crawls.
if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed — GPU driver not ready. Do NOT start training yet."
  exit 1
fi
echo "GPU OK:"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

# 1) Resume the main queue (#1 done already; continues #5 / #2, skipping finished).
nohup ./tstr_resume.sh >/dev/null 2>&1 &
echo "  launched: tstr_resume.sh (main queue)"

# 2) Re-arm the topbrain seed-1 fixer (waits for the queue, then re-runs + aggregates).
nohup ./tstr_fix_topbrain_seed1.sh >/dev/null 2>&1 &
echo "  launched: tstr_fix_topbrain_seed1.sh (waits, then fixes + aggregates)"

echo "Resumed. Progress: tstr_resume.log / tstr_fix_topbrain_seed1.log"
