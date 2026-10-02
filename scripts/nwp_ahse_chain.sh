#!/usr/bin/env bash
# Full AHSE pipeline with and without GFS covariates on the same period (end of the GFS
# archive), 1 h resolution, seed 42.
#   A: results/yulara_neighbours_2018      no GFS
#   B: results/yulara_neighbours_2018_gfs  GFS covariates for the tree models + GFS member
# The deep-learning members do not use GFS: B reuses A's trained checkpoints so that the two
# runs differ only by the GFS information.
cd "$(dirname "$0")/.." || exit 1
PY=/home/billciss/miniconda3/bin/python
CFG=configs/data_config_yulara_neighbours.yaml
DATA=data/processed/yulara_neighbours_1h.csv
END=2018-12-31
A=results/yulara_neighbours_2018
B=results/yulara_neighbours_2018_gfs
LOG=results/overnight/nwp_ahse_chain.log
mkdir -p results/overnight "$A" "$B/models"

while pgrep -f nwp_experiment.py > /dev/null; do sleep 60; done   # free the CPU first

echo "$(date '+%F %T') A (no GFS) start" >> "$LOG"
$PY main.py --horizon 1h --models all --seed 42 --data-config $CFG --data-file $DATA --end $END \
    --results-dir $A > $A/run_1h.log 2>&1 || { echo "$(date '+%F %T') A FAILED" >> "$LOG"; exit 1; }
echo "$(date '+%F %T') A done" >> "$LOG"

cp $A/models/{PatchTST,N-HiTS,LSTM,GRU}_1h.pt $B/models/ \
    || { echo "$(date '+%F %T') checkpoint copy FAILED" >> "$LOG"; exit 1; }

echo "$(date '+%F %T') B (GFS) start" >> "$LOG"
$PY main.py --horizon 1h --models all --seed 42 --data-config $CFG --data-file $DATA --end $END \
    --gfs data_external/gfs/gfs_yulara_DSWRF.csv data_external/gfs/gfs_yulara_TCDC.csv \
    --results-dir $B > $B/run_1h.log 2>&1 || { echo "$(date '+%F %T') B FAILED" >> "$LOG"; exit 1; }
echo "$(date '+%F %T') B done" >> "$LOG"
touch results/overnight/NWP_AHSE_DONE
