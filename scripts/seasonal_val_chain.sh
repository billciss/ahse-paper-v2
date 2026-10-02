#!/usr/bin/env bash
# Season-balanced (interleaved) validation, with and without GFS, NIST then Yulara (1 h, seed 42).
# The test set is the same final block as with chronological validation, so every run is compared
# with its chronological counterpart on identical test forecasts.
cd "$(dirname "$0")/.." || exit 1
PY=/home/billciss/miniconda3/bin/python
G=data_external/gfs
LOG=results/overnight/seasonal_val.log
mkdir -p results/overnight
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

run_site() {  # name data config gfs_prefix chrono_A chrono_B
  local name=$1 data=$2 cfg=$3 gfs=$4 chA=$5 chB=$6
  local A=results/${name}_seasonal B=results/${name}_seasonal_gfs
  mkdir -p $A $B/models
  say "$name A (seasonal, no GFS) start"
  $PY main.py --horizon 1h --models all --seed 42 --data-config $cfg --data-file $data \
      --results-dir $A > $A/run_1h.log 2>&1 || { say "$name A FAILED"; return 1; }
  cp $A/models/{PatchTST,N-HiTS,LSTM,GRU}_1h.pt $B/models/ || { say "$name checkpoint copy FAILED"; return 1; }
  say "$name B (seasonal, GFS) start"
  $PY main.py --horizon 1h --models all --seed 42 --data-config $cfg --data-file $data \
      --gfs $G/gfs_${gfs}_DSWRF.csv $G/gfs_${gfs}_TCDC.csv --results-dir $B > $B/run_1h.log 2>&1 \
      || { say "$name B FAILED"; return 1; }
  $PY scripts/compare_nwp_runs.py --a $A --b $B --out $B/tables/comparison_no_gfs_vs_gfs > $B/compare.log 2>&1
  $PY scripts/compare_nwp_runs.py --a $chA --b $A --out $A/tables/comparison_chrono_vs_seasonal > $A/compare_chrono.log 2>&1
  $PY scripts/compare_nwp_runs.py --a $chB --b $B --out $B/tables/comparison_chrono_vs_seasonal > $B/compare_chrono.log 2>&1
  $PY scripts/conformal_intervals.py --cache $B/tables/predictions_1h.npz --horizon 1h --out $B/tables \
      --key-model GFS > $B/conformal.log 2>&1
  say "$name done"
}

run_site nist data/processed/nist_pvdaq_1h.csv configs/data_config_nist_interleaved.yaml nist \
    results/nist results/nist_gfs || exit 1
run_site yulara_neighbours data/processed/yulara_neighbours_1h.csv \
    configs/data_config_yulara_neighbours_interleaved.yaml yulara \
    results/yulara_neighbours results/yulara_neighbours_gfs || exit 1
say "all done"
touch results/overnight/SEASONAL_DONE
