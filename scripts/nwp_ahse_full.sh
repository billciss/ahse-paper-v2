#!/usr/bin/env bash
# Full-period AHSE run with GFS covariates (Yulara uncurtailed sites, 1 h, seed 42), compared with
# the existing run without GFS (results/yulara_neighbours). Waits until the Yulara GFS archive
# covers every month of the series, reuses the deep-learning checkpoints of the run without GFS
# (they receive no GFS input), retrains the tree models with GFS, then runs the comparison and
# the conformal intervals.
cd "$(dirname "$0")/.." || exit 1
PY=/home/billciss/miniconda3/bin/python
CFG=configs/data_config_yulara_neighbours.yaml
DATA=data/processed/yulara_neighbours_1h.csv
A=results/yulara_neighbours
B=results/yulara_neighbours_gfs
LOG=results/overnight/nwp_ahse_full.log
G=data_external/gfs
mkdir -p results/overnight "$B/models"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

complete() {  # every month 2016-04 .. 2020-05 done for both parameters
  $PY - <<'EOF'
import json, sys
import pandas as pd
months = [str(p) for p in pd.period_range("2016-04", "2020-05", freq="M")]
for p in ("DSWRF", "TCDC"):
    s = json.load(open(f"data_external/gfs/state_yulara_{p}.json"))
    if any((s.get(m) or {}).get("status") != "done" for m in months):
        sys.exit(1)
EOF
}
say "waiting for a complete Yulara GFS archive"
until complete; do sleep 300; done
while pgrep -f "gdex_batch.py --site yulara" > /dev/null; do sleep 60; done   # last append finished
say "archive complete"

cp $A/models/{PatchTST,N-HiTS,LSTM,GRU}_1h.pt $B/models/ || { say "checkpoint copy FAILED"; exit 1; }
say "B (GFS, full period) start"
$PY main.py --horizon 1h --models all --seed 42 --data-config $CFG --data-file $DATA \
    --gfs $G/gfs_yulara_DSWRF.csv $G/gfs_yulara_TCDC.csv --results-dir $B > $B/run_1h.log 2>&1 \
    || { say "B FAILED"; exit 1; }
say "B done"
$PY scripts/compare_nwp_runs.py --a $A --b $B --out $B/tables/comparison_no_gfs_vs_gfs > $B/compare.log 2>&1 \
    || { say "comparison FAILED"; exit 1; }
$PY scripts/conformal_intervals.py --cache $B/tables/predictions_1h.npz --horizon 1h --out $B/tables \
    --key-model GFS > $B/conformal.log 2>&1 || { say "conformal FAILED"; exit 1; }
say "all done"
touch results/overnight/NWP_FULL_DONE
