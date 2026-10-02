"""
yulara_data_quality.py
======================
Data-quality scan of the Yulara per-site 5-min series, by site and year, on daytime
samples (measured GHI > 300 W/m2):
  missing      : share of NaN power
  zero_output  : share with power < 1% of nameplate (outage / trip / disconnection)
  stuck        : share inside runs of >= 2 h of identical power values (frozen logger)
  low_vs_peers : share where the site's clear-sky index is < 30% of the median index
                 of the other uncurtailed sites (partial outage), neighbours only
Also lists the longest daytime outages per site.

Usage (WSL):
    python scripts/yulara_data_quality.py --downloads /mnt/c/Users/billc/Downloads \
        --out results/curtailment/site_data_quality.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yulara_sites_analysis import SITES, read_power  # noqa: E402

SITE_KEYS = ["DesertGardens", "ServiceStation", "SailsCombined", "Laundry", "ConnellanAirport", "Total"]
NEIGHBOURS = ["ServiceStation", "SailsCombined", "Laundry", "ConnellanAirport"]


def run_lengths(mask: pd.Series) -> pd.Series:
    """Length (in samples) of the run each True element belongs to."""
    grp = (mask != mask.shift()).cumsum()
    return mask.groupby(grp).transform("sum").where(mask, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    P = pd.DataFrame({k: read_power(args.downloads / SITES[k][0]) for k in SITE_KEYS})
    ghi = pd.read_csv(args.downloads / SITES["Total"][0], index_col=0,
                      usecols=["timestamp", "Global_Horizontal_Radiation"])["Global_Horizontal_Radiation"]
    ghi.index = pd.to_datetime(ghi.index)
    ghi = ghi[~ghi.index.duplicated()].reindex(P.index)
    day = ghi > 300

    # per-site "clear-sky index" proxy: power / (nameplate * GHI/1000)
    kc = pd.DataFrame({k: P[k] / (SITES[k][1] * ghi / 1000) for k in SITE_KEYS})
    rows = []
    for k in SITE_KEYS:
        p = P[k]
        stuck = run_lengths((p.diff() == 0) & day) >= 24          # >= 2 h of 5-min samples
        zero = p < 0.01 * SITES[k][1]
        if k in NEIGHBOURS:
            peers = kc[[n for n in NEIGHBOURS if n != k]].median(axis=1)
            low = kc[k] < 0.3 * peers
        else:
            low = pd.Series(False, index=p.index)
        for yr, idx in p[day].groupby(p[day].index.year).groups.items():
            rows.append({"site": k, "year": int(yr), "daytime_n": len(idx),
                         "missing": round(float(p.loc[idx].isna().mean()), 4),
                         "zero_output": round(float(zero.loc[idx].mean()), 4),
                         "stuck": round(float(stuck.loc[idx].mean()), 4),
                         "low_vs_peers": round(float(low.loc[idx].mean()), 4)})
    q = pd.DataFrame(rows)
    flagged = q[(q[["missing", "zero_output", "stuck", "low_vs_peers"]] > 0.02).any(axis=1)]
    print("site-years with any indicator > 2% of daytime samples:")
    print(flagged.to_string(index=False) if len(flagged) else "   none")

    print("\nlongest daytime zero-output episodes (days with >= 50% zero daytime output):")
    for k in SITE_KEYS:
        z = ((P[k] < 0.01 * SITES[k][1]) | P[k].isna())[day]
        daily = z.groupby(z.index.date).mean()
        bad = daily[daily >= 0.5]
        if len(bad):
            d = pd.to_datetime(pd.Series(bad.index))
            runs = (d.diff() != pd.Timedelta("1D")).cumsum()
            spans = d.groupby(runs).agg(["min", "max", "count"]).sort_values("count", ascending=False).head(3)
            print(f"   {k:17s} {len(bad):4d} days;  longest: "
                  + "; ".join(f"{r['min'].date()}->{r['max'].date()} ({r['count']} d)" for _, r in spans.iterrows()))
        else:
            print(f"   {k:17s}    0 days")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        q.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
