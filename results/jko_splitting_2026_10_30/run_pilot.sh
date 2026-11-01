#!/bin/bash
cd /ai/saisail/jko-splitting/repo
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u +%FT%TZ > ../pilot.start
timeout 58m nice -n 19 .venv/bin/python experiments/jko_splitting_2026_10_30.py pilot --workers 4 > ../pilot.log 2>&1
echo "rc=$?" >> ../pilot.log
date -u +%FT%TZ > ../pilot.end
