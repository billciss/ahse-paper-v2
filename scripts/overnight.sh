#!/usr/bin/env bash
# Overnight chain (run from the repository root in WSL, detached with nohup).
#   GPU chain : wait for the running 15-min job -> rerun it from checkpoints (block AHSE +
#               prediction cache) -> per-horizon report at 15 min -> 4 extra seeds at 1 h
#               (neural models retrained, GBDT checkpoints reused) -> seed summary.
#   GFS chain : wait for the Yulara GDEX batches -> NIST batches (2016-2018, purge on).
# Every step logs to results/overnight/; a step failure is logged and the chain continues.
set -u
PY=/home/billciss/miniconda3/bin/python
CFG=configs/data_config_yulara_neighbours.yaml
LOG=results/overnight
mkdir -p "$LOG"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG/overnight.log"; }
wait_for() { while pgrep -f "$1" > /dev/null; do sleep 60; done; }

gpu_chain() {
  say "GPU: waiting for the running 15-min job"
  wait_for "main.py --horizon 15min"
  say "GPU: 15-min rerun from checkpoints (block AHSE + cache)"
  $PY main.py --horizon 15min --models all --data-config $CFG --results-dir results/yulara_neighbours \
      > "$LOG/run_15min_blocks.log" 2>&1 || say "GPU: 15-min rerun FAILED (see log)"
  say "GPU: per-horizon report 15 min"
  $PY scripts/per_horizon_selection.py --cache results/yulara_neighbours/tables/predictions_15min.npz \
      --steps-per-hour 4 --out results/yulara_neighbours/tables/per_horizon_selection_15min.json \
      > "$LOG/per_horizon_15min.log" 2>&1 || say "GPU: 15-min report FAILED"
  for s in 1 2 3 4; do
    d=results/seeds/seed_$s/yulara_neighbours
    mkdir -p "$d/models"
    cp -n results/yulara_neighbours/models/LightGBM_1h.joblib results/yulara_neighbours/models/XGBoost_1h.joblib "$d/models/"
    say "GPU: seed $s at 1 h"
    $PY main.py --horizon 1h --models all --seed $s --data-config $CFG --results-dir "$d" \
        > "$LOG/run_1h_seed_$s.log" 2>&1 || say "GPU: seed $s FAILED (see log)"
  done
  say "GPU: seed summary"
  $PY scripts/summarize_seeds.py --horizon 1h --out results/seeds/summary_1h.csv \
      --runs results/yulara_neighbours results/seeds/seed_*/yulara_neighbours > "$LOG/seed_summary_1h.log" 2>&1
  say "GPU: chain finished"
}

gfs_chain() {
  say "GFS: waiting for the Yulara batches"
  wait_for "gdex_batch.py --site yulara"
  say "GFS: NIST batches"
  $PY scripts/gdex_batch.py --site nist --param DSWRF --start 2016-01 --end 2018-12 --max-active 3 \
      --purge-after-download > data_external/gfs/batch_nist_DSWRF.log 2>&1 &
  $PY scripts/gdex_batch.py --site nist --param "T CDC" --start 2016-01 --end 2018-12 --max-active 3 \
      --purge-after-download > data_external/gfs/batch_nist_TCDC.log 2>&1 &
  wait
  say "GFS: chain finished"
}

gpu_chain &
gfs_chain &
wait
say "ALL DONE"
touch "$LOG/ALL_DONE"
