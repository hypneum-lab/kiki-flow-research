#!/bin/bash
# Splitting confirmatory run (prereg-jko-splitting-20261030, deviation: host macM3).
# Waits for (a) the M5 run: last "start" of main.log followed by an "end", and
# (b) the end of the NH-03 confirmatory: its waiter (pid 1878) gone, no NH-03
# worker or merge process, and a terminal line in its wait.log. Two consecutive
# free checks 5 min apart, then launches ONCE: 4 workers, nice 19, CPU only.
set -u
BASE=$HOME/scratch/jko-splitting
REPO=$BASE/repo
TAG_COMMIT=09ec14466b44f41061f9d49194053050f4cdb6bf
M5LOG=$HOME/scratch/m5-macm3/logs/main.log
NH03LOG=$HOME/scratch/nh03-deferred/confirm/wait.log
NH03PID=1878
GO="PI 2026-10-30 15:19 CET, QCM, session electron-fb"
LOG=$BASE/wait.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
m5_done() { awk "/\\] start/{st=1; en=0} /\\] end/{if (st) en=1} END{exit !(st && en)}" "$M5LOG"; }
nh03_done() {
  ! kill -0 $NH03PID 2>/dev/null \
  && ! pgrep -f "cpu_wrap.py|run_nh03|wait_and_run.sh" >/dev/null \
  && grep -qE "workers exit|NOT launching|ceiling reached|abort" "$NH03LOG"
}
log "waiting (pid $$)"
ok=0
while :; do
  if m5_done && nh03_done; then ok=$((ok+1)); else ok=0; fi
  log "check: m5_done=$(m5_done && echo 1 || echo 0) nh03_done=$(nh03_done && echo 1 || echo 0) consecutive_free=$ok"
  [ $ok -ge 2 ] && break
  sleep 300
done
cd "$REPO" || exit 1
if [ "$(git rev-parse HEAD)" != "$TAG_COMMIT" ] || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "repo not at clean tag commit; NOT launching"; exit 3
fi
[ -e "$BASE/launched" ] && { log "already launched once; abort"; exit 4; }
touch "$BASE/launched"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
date -u +%FT%TZ > "$BASE/confirm.start"
log "launch confirm: 4 workers, nice 19, commit $TAG_COMMIT"
caffeinate -is nice -n 19 .venv/bin/python experiments/jko_splitting_2026_10_30.py confirm --workers 4 --go "$GO" > "$BASE/confirm.log" 2>&1
rc=$?
date -u +%FT%TZ > "$BASE/confirm.end"
log "confirm exit $rc"
nice -n 19 .venv/bin/python experiments/jko_splitting_2026_10_30.py analyse > "$BASE/analyse.log" 2>&1
log "analyse exit $?"
