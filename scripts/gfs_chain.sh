#!/usr/bin/env bash
# Resumes the GDEX GFS extraction: Yulara (both parameters, from their saved state), then NIST.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/billciss/miniconda3/bin/python
G=data_external/gfs
say() { echo "$(date '+%F %T') $*" | tee -a results/overnight/overnight.log; }
run_site() {  # site start end
  $PY scripts/gdex_batch.py --site "$1" --param DSWRF --start "$2" --end "$3" --max-active 3 \
      --purge-after-download >> "$G/batch_$1_DSWRF.log" 2>&1 &
  $PY scripts/gdex_batch.py --site "$1" --param "T CDC" --start "$2" --end "$3" --max-active 3 \
      --purge-after-download >> "$G/batch_$1_TCDC.log" 2>&1 &
  wait
}
say "GFS: resume Yulara"
run_site yulara 2016-04 2020-05
say "GFS: NIST"
run_site nist 2016-01 2018-12
say "GFS: chain finished"
touch results/overnight/GFS_DONE
