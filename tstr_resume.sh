#!/bin/bash
# Resilient TSTR sweep driver. Skips any run whose best.pt already exists, so it
# is safe to re-launch after an interruption (e.g. reboot). Covers:
#   #1 main 3-seed sweep (synth-only + real)
#   #5 real+synth augmentation (3 seeds)
#   #2 ResUNet robustness (seed 0, synth-only)
cd /home/marija/Desktop/ferit/ROBOMED/godina1/wsi_app
PY=/home/marija/anaconda3/bin/python
LOG=tstr_resume.log
: > "$LOG"

run() {  # dataset mode seed [arch]
  local ds=$1 mode=$2 seed=$3 arch=${4:-unet}
  local archsuf=""; archarg=""
  if [ "$arch" = "resunet" ]; then archsuf="_resunet"; archarg="--arch resunet"; fi
  local dir="tstr_synth_models/${ds}_${mode}${archsuf}_s${seed}"
  if [ -f "$dir/best.pt" ]; then echo "[skip] $dir" >>"$LOG"; return; fi
  echo "════ $(date +%H:%M:%S) ${ds}/${mode}${archsuf} seed=${seed} ════" >>"$LOG"
  $PY tstr_synth.py train --dataset "$ds" --mode "$mode" --seed "$seed" \
      --seeded_dir $archarg >>"$LOG" 2>&1
  echo "[ok] $dir done $(date +%H:%M:%S)" >>"$LOG"
}

PROS_SYNTH="synth_combined synth_cyclegan synth_ldm synth_pix2pix"
TOP_SYNTH="synth_combined synth_cyclegan synth_ldm synth_pix2pix"
PASD_SYNTH="synth_ldm synth_pix2pix synth_spade"

# ── #1 main sweep: 3 seeds ──
for seed in 0 1 2; do
  for m in real $PROS_SYNTH; do run prostate "$m" $seed; done
  for m in real $TOP_SYNTH;  do run topbrain "$m" $seed; done
  for m in real $PASD_SYNTH; do run pasd     "$m" $seed; done
done
echo "──── #1 main sweep complete $(date) ────" >>"$LOG"

# ── #5 augmentation (real+synth): 3 seeds ──
for seed in 0 1 2; do
  for m in real_synth_combined real_synth_cyclegan real_synth_ldm real_synth_pix2pix; do run prostate "$m" $seed; done
  for m in real_synth_combined real_synth_cyclegan real_synth_ldm real_synth_pix2pix; do run topbrain "$m" $seed; done
  for m in real_synth_ldm real_synth_pix2pix real_synth_spade; do run pasd "$m" $seed; done
done
echo "──── #5 augmentation complete $(date) ────" >>"$LOG"

# ── #2 ResUNet robustness: seed 0, synth-only ──
for m in $PROS_SYNTH; do run prostate "$m" 0 resunet; done
for m in $TOP_SYNTH;  do run topbrain "$m" 0 resunet; done
for m in $PASD_SYNTH; do run pasd     "$m" 0 resunet; done
echo "──── #2 ResUNet complete $(date) ────" >>"$LOG"

echo "ALL TSTR RUNS DONE $(date)" >>"$LOG"
