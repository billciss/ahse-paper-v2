#!/usr/bin/env bash
# NIST Ground array from PVDAQ (system 4902), 2016-01-01 to 2017-10-10 (the horizontal
# pyranometer changes on 2017-10-1x with an unknown constant, see convert_pvdaq_nist.py), 1 h, seed 42:
# conversion + preparation, AHSE without and with GFS, comparison, conformal intervals.
# Waits for the seed study (CPU) and for the NIST GFS months up to 2018-03.
cd "$(dirname "$0")/.." || exit 1
PY=/home/billciss/miniconda3/bin/python
CFG=configs/data_config_nist.yaml
G=data_external/gfs
A=results/nist
B=results/nist_gfs
LOG=results/overnight/nwp_nist.log
mkdir -p results/overnight $A $B/models
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "conversion and preparation"
rm -f data_external/nist_pvdaq/Ground/*.csv.gz
$PY scripts/convert_pvdaq_nist.py --pvdaq data_external/pvdaq_4902 \
    --nist-zip data_external/nist/Ground/onemin-Ground-2016.zip --out data_external/nist_pvdaq/Ground \
    > $A/convert.log 2>&1 || { say "conversion FAILED"; exit 1; }
$PY scripts/prepare_nist.py --raw-dir data_external/nist_pvdaq --array Ground --weather-source array \
    --start 2016-01-01 --end 2017-10-10 --tag nist_pvdaq > $A/prepare.log 2>&1 || { say "preparation FAILED"; exit 1; }
DATA=data/processed/nist_pvdaq_1h.csv

gfs_ready() {
  $PY - <<'EOF'
import json, sys
import pandas as pd
months = [str(p) for p in pd.period_range("2016-01", "2017-10", freq="M")]
for p in ("DSWRF", "TCDC"):
    try:
        s = json.load(open(f"data_external/gfs/state_nist_{p}.json"))
    except FileNotFoundError:
        sys.exit(1)
    if any((s.get(m) or {}).get("status") != "done" for m in months):
        sys.exit(1)
EOF
}
say "waiting for the seed study and the NIST GFS months"
until [ -f results/overnight/NWP_SEEDS_DONE ] && gfs_ready; do sleep 300; done

say "A (no GFS) start"
$PY main.py --horizon 1h --models all --seed 42 --data-config $CFG --data-file $DATA \
    --results-dir $A > $A/run_1h.log 2>&1 || { say "A FAILED"; exit 1; }
say "A done"
cp $A/models/{PatchTST,N-HiTS,LSTM,GRU}_1h.pt $B/models/ || { say "checkpoint copy FAILED"; exit 1; }
say "B (GFS) start"
$PY main.py --horizon 1h --models all --seed 42 --data-config $CFG --data-file $DATA \
    --gfs $G/gfs_nist_DSWRF.csv $G/gfs_nist_TCDC.csv --results-dir $B > $B/run_1h.log 2>&1 \
    || { say "B FAILED"; exit 1; }
say "B done"
$PY scripts/compare_nwp_runs.py --a $A --b $B --out $B/tables/comparison_no_gfs_vs_gfs > $B/compare.log 2>&1 \
    || { say "comparison FAILED"; exit 1; }
$PY scripts/conformal_intervals.py --cache $B/tables/predictions_1h.npz --horizon 1h --out $B/tables \
    --key-model GFS > $B/conformal.log 2>&1 || { say "conformal FAILED"; exit 1; }
say "all done"
touch results/overnight/NIST_DONE
