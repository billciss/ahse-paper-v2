#!/usr/bin/env bash
# Seed study with and without GFS (Yulara uncurtailed, 1 h, full period).
# For each seed s in 1..4 the deep-learning members are the checkpoints already trained with
# seed s (results/seeds/seed_s), and the tree models are retrained with seed s:
#   results/seeds_gfs/seed_s/no_gfs   without GFS
#   results/seeds_gfs/seed_s/gfs      with GFS covariates and the GFS member
# Seed 42 is results/yulara_neighbours (no GFS) and results/yulara_neighbours_gfs (GFS), whose
# trees already use seed 42. Then: per-arm summaries and paired comparisons per seed.
cd "$(dirname "$0")/.." || exit 1
PY=/home/billciss/miniconda3/bin/python
CFG=configs/data_config_yulara_neighbours.yaml
DATA=data/processed/yulara_neighbours_1h.csv
G=data_external/gfs
OUT=results/seeds_gfs
LOG=results/overnight/nwp_seeds.log
mkdir -p results/overnight $OUT
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

for s in 1 2 3 4; do
  for arm in no_gfs gfs; do
    d=$OUT/seed_$s/$arm
    mkdir -p $d/models
    cp results/seeds/seed_$s/yulara_neighbours/models/{PatchTST,N-HiTS,LSTM,GRU}_1h.pt $d/models/ \
        || { say "seed $s $arm: checkpoint copy FAILED"; exit 1; }
    extra=()
    [ "$arm" = gfs ] && extra=(--gfs $G/gfs_yulara_DSWRF.csv $G/gfs_yulara_TCDC.csv)
    say "seed $s $arm start"
    $PY main.py --horizon 1h --models all --seed $s --data-config $CFG --data-file $DATA \
        "${extra[@]}" --results-dir $d > $d/run_1h.log 2>&1 || { say "seed $s $arm FAILED"; exit 1; }
    say "seed $s $arm done"
  done
  $PY scripts/compare_nwp_runs.py --a $OUT/seed_$s/no_gfs --b $OUT/seed_$s/gfs \
      --out $OUT/seed_$s/comparison_no_gfs_vs_gfs > $OUT/seed_$s/compare.log 2>&1 \
      || { say "seed $s comparison FAILED"; exit 1; }
done

$PY scripts/summarize_seeds.py --horizon 1h --out $OUT/summary_no_gfs_1h.csv \
    --runs results/yulara_neighbours $OUT/seed_*/no_gfs > $OUT/summary_no_gfs.log 2>&1
$PY scripts/summarize_seeds.py --horizon 1h --out $OUT/summary_gfs_1h.csv \
    --runs results/yulara_neighbours_gfs $OUT/seed_*/gfs > $OUT/summary_gfs.log 2>&1
say "all done"
touch results/overnight/NWP_SEEDS_DONE
