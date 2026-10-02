"""
fetch_pvdaq.py
==============
Downloads daily PVDAQ parquet files of one system from the public OEDI data lake (NREL,
doi:10.25984/1846021, CC-BY 4.0; no account needed), skipping files already on disk.

    https://oedi-data-lake.s3.amazonaws.com/pvdaq/parquet/pvdata/system_id=<id>/year=Y/month=M/day=D/*.parquet

Usage (WSL):
    python scripts/fetch_pvdaq.py --system 4902 --start 2016-01 --end 2018-03 --out data_external/pvdaq_4902
"""

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BUCKET = "https://oedi-data-lake.s3.amazonaws.com"


def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=120) as r:
            xml = r.read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(key, dest, retries=5):
    for k in range(retries):
        try:
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", tmp)
            tmp.replace(dest)
            return
        except OSError:
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"download failed: {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", type=int, required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    total, n_new = 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        for p in pd.period_range(args.start, args.end, freq="M"):
            prefix = f"pvdaq/parquet/pvdata/system_id={args.system}/year={p.year}/month={p.month}/"
            keys = list_keys(prefix)
            folder = args.out / f"{p.year}" / f"{p.month:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            todo = [(k, folder / Path(k).name) for k in keys if not (folder / Path(k).name).exists()]
            list(pool.map(lambda kd: fetch(*kd), todo))
            n_new += len(todo)
            total += sum((folder / Path(k).name).stat().st_size for k in keys)
            print(f"{p}: {len(keys)} files", flush=True)
    print(f"done: {n_new} new files, {total / 1e6:.1f} MB on disk")


if __name__ == "__main__":
    main()
