#!/bin/bash
# Relaunch of the splitting confirmatory run after the os.nice fix
# (deviation note docs/notes/2026-10-31-jko-splitting-osnice-deviation.md).
set -u
BASE=$HOME/scratch/jko-splitting
REPO=$BASE/repo
COMMIT=674118177a4ccb5d6ec294d7d787f64805624b38
GO="PI 2026-10-30 15:19 CET, QCM, session electron-fb"
LOG=$BASE/wait.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cd "$REPO" || exit 1
if [ "$(git rev-parse HEAD)" != "$COMMIT" ] || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "relaunch: repo not at clean $COMMIT; NOT launching"; exit 3
fi
[ -e "$BASE/relaunched" ] && { log "relaunch: already launched once; abort"; exit 4; }
[ -e results/jko_splitting_2026_10_30/confirm.jsonl ] && { log "relaunch: confirm.jsonl exists; abort"; exit 5; }
touch "$BASE/relaunched"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
date -u +%FT%TZ > "$BASE/confirm.start"
log "relaunch confirm: 4 workers, nice 19, commit $COMMIT"
caffeinate -is nice -n 19 .venv/bin/python experiments/jko_splitting_2026_10_30.py confirm --workers 4 --go "$GO" > "$BASE/confirm.log" 2>&1
rc=$?
date -u +%FT%TZ > "$BASE/confirm.end"
log "confirm exit $rc"
nice -n 19 .venv/bin/python experiments/jko_splitting_2026_10_30.py analyse > "$BASE/analyse.log" 2>&1
log "analyse exit $?"
