#!/usr/bin/env bash
# GFS extraction for the NIST site (2016-2018), both parameters in parallel, resuming from the
# saved state; returns when both batches have finished.
cd "$(dirname "$0")/.." || exit 1
PY=/home/billciss/miniconda3/bin/python
G=data_external/gfs
$PY scripts/gdex_batch.py --site nist --param DSWRF --start 2016-01 --end 2018-12 --max-active 3 \
    --purge-after-download >> $G/batch_nist_DSWRF.log 2>&1 &
$PY scripts/gdex_batch.py --site nist --param "T CDC" --start 2016-01 --end 2018-12 --max-active 3 \
    --purge-after-download >> $G/batch_nist_TCDC.log 2>&1 &
wait
tail -n 2 $G/batch_nist_DSWRF.log $G/batch_nist_TCDC.log
