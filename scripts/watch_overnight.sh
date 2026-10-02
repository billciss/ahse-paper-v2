#!/usr/bin/env bash
# Exits (so that the caller is notified) as soon as the overnight chain reports a new problem
# or finishes. Problems: new "FAILED" lines in overnight.log, new Tracebacks or failed months
# in the GDEX batch logs, or a job log containing a Traceback.
cd "$(dirname "$0")/.." || exit 1
count() { cat "$@" 2>/dev/null | grep -E "$PATTERN" | grep -v -c "failed months: none" ; }
PATTERN="FAILED|Traceback| failed|archive covers|GDEX API error|unreachable"
LOGS="results/overnight/overnight.log data_external/gfs/batch_*.log results/overnight/run_*.log results/overnight/nwp_*.log"
base=$(count $LOGS)
while true; do
  if [ -f "results/overnight/${1:-ALL_DONE}" ]; then echo "${1:-ALL_DONE}"; exit 0; fi
  now=$(count $LOGS)
  if [ "$now" -gt "$base" ]; then
    echo "NEW PROBLEM ($base -> $now)"
    grep -n -E "$PATTERN" $LOGS 2>/dev/null | grep -v "failed months: none" | tail -8
    exit 2
  fi
  sleep 300
done
