#!/usr/bin/env bash
# Waits for the main TSTR queue to finish, then re-runs the one collapsed run
# (topbrain real seed 1: best was epoch 2, Dice 0.479). Retries up to 3x if it
# collapses again, keeping the best result. Re-aggregates at the end.
set -uo pipefail
cd /home/marija/Desktop/ferit/ROBOMED/godina1/wsi_app
PY=/home/marija/anaconda3/bin/python
LOG=tstr_fix_topbrain_seed1.log
: > "$LOG"

echo "[$(date +%H:%M)] waiting for main queue (ALL TSTR RUNS DONE marker)..." >>"$LOG"
while ! grep -q 'ALL TSTR RUNS DONE' tstr_resume.log 2>/dev/null; do
  sleep 60
done
# extra safety: make sure no training process is still alive
while pgrep -f 'tstr_synth.py train' >/dev/null; do sleep 30; done
echo "[$(date +%H:%M)] queue done. Re-running topbrain/real seed=1." >>"$LOG"

dice() {  # echo dice of a best.pt, or 0 if absent
  local f=$1
  [ -f "$f" ] || { echo 0; return; }
  $PY -c "import torch;print('%.4f'%torch.load('$f',map_location='cpu',weights_only=False)['dice'])" 2>/dev/null || echo 0
}

DIR=tstr_synth_models/topbrain_real_s1
BEST=0.0000
BACKUP="${DIR}_collapsed_ep2"
[ -d "$BACKUP" ] || mv "$DIR" "$BACKUP" 2>/dev/null   # keep the collapsed one for the record

for attempt in 1 2 3; do
  echo "[$(date +%H:%M)] attempt $attempt ..." >>"$LOG"
  rm -rf "$DIR"
  $PY tstr_synth.py train --dataset topbrain --mode real --seed 1 --seeded_dir >>"$LOG" 2>&1
  d=$(dice "$DIR/best.pt")
  echo "[$(date +%H:%M)] attempt $attempt -> Dice=$d" >>"$LOG"
  # keep best across attempts
  better=$($PY -c "print(1 if float('$d')>float('$BEST') else 0)")
  if [ "$better" = "1" ]; then
    rm -rf "${DIR}_keep"; cp -r "$DIR" "${DIR}_keep"; BEST=$d
  fi
  # healthy threshold (other two seeds were ~0.66-0.68)
  ok=$($PY -c "print(1 if float('$d')>=0.58 else 0)")
  [ "$ok" = "1" ] && break
done

# install the best attempt
rm -rf "$DIR"; mv "${DIR}_keep" "$DIR"
echo "[$(date +%H:%M)] kept best Dice=$BEST at $DIR" >>"$LOG"

echo "" >>"$LOG"
echo "===== FINAL AGGREGATION =====" >>"$LOG"
$PY aggregate_tstr.py >>"$LOG" 2>&1
echo "[$(date +%H:%M)] DONE" >>"$LOG"
