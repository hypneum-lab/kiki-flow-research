#!/bin/bash
# M5 certify, run 2 (PI decision 2026-10-30 13:57 CET: macM3 finishes the whole grid).
# Waits for the NH-81 energy measurement to finish (+2 min), then runs the 37 remaining configs.
D=/Users/electron/scratch/m5-macm3
cd "$D" || exit 1
export MLX_DISABLE_COMPILE=1 PYTHONPATH=$D/repo:$D/repo/experiments
W="$D/logs/wait_nh81.log"
echo "$(date -u +%FT%TZ) waiting for NH-81 (pgrep nh81-m3)" >> "$W"
free=0
while [ $free -lt 2 ]; do
  if pgrep -f "nh81-m3" >/dev/null; then free=0; else free=$((free+1)); fi
  sleep 60
done
echo "$(date -u +%FT%TZ) NH-81 gone; extra 120 s" >> "$W"
sleep 120
echo "[$(date -u +%FT%TZ)] start" >> "$D/logs/main.log"
caffeinate -is nice -n 19 "$D/repo/.venv/bin/python" "$D/certify_shard.py" --split 0 --workers 4 \
  --out "$D/results" --split-file "$D/split_v2.txt" \
  --rule-sha 61dd69e7ac37f357caddb0b1c539e3b16224a040 \
  --exclude-jsonl "$D/es_certify_13.jsonl" >> "$D/logs/certify_shard_v2.log" 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] end exit=$rc" >> "$D/logs/main.log"
