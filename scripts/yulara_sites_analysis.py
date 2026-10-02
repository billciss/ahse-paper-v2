"""
yulara_sites_analysis.py
========================
Per-site analysis of the Yulara (Ayers Rock Resort, off-grid) PV system from the DKASC
per-site 5-min downloads:
  1. consistency: do the sub-sites add up to the "Total of all sites" series?
  2. which series was used in the AHSE paper (data/raw/data_5min.csv)?
  3. apparent curtailment per site (same clear-sky method as quantify_curtailment.py)

Usage:
    python scripts/yulara_sites_analysis.py --downloads /mnt/c/Users/billc/Downloads \
        --paper-series data/raw/data_5min.csv --out-dir results/curtailment
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantify_curtailment import RAW_DKASC_COLUMNS, TZ, analyse  # noqa: E402

SITES = {  # key: (file name, nameplate kW)
    "DesertGardens": ("5-Site_DG-PV1-DB-DG-M1A.csv", 1058.4),
    "ServiceStation": ("10-Site_SS-PV1-DB-SS-1A.csv", 226.8),
    "ConnellanAirport": ("8-Site_CA-PV1-DB-CA-1A.csv", 105.9),
    "Sails1": ("7-Site_SD-PV2-DB-SD-1A.csv", 22.6),
    "Sails2": ("9-Site_SD-PV2-DB-SD-2A.csv", 38.3),
    "Sails3": ("6-Site_SD-PV2-DB-SD-3A.csv", 45.8),
    "SailsCombined": ("3051-Site_Total_Site-Sails_in_the_Desert.csv", 106.6),
    "Laundry": ("11-Site_LD-PV1-DB-LD-1A.csv", 327.6),
    "Total": ("3050-Site_Total_Site-PV_Generation.csv", 1820.0),
}
REFERENCE = ["ServiceStation", "ConnellanAirport", "SailsCombined", "Laundry"]


def read_power(path):
    s = pd.read_csv(path, usecols=["timestamp", "Active_Power"], index_col=0)["Active_Power"]
    s.index = pd.to_datetime(s.index)
    return s[~s.index.duplicated()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path, required=True)
    ap.add_argument("--paper-series", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    P = pd.DataFrame({k: read_power(args.downloads / f) for k, (f, _) in SITES.items()})
    weather = pd.read_csv(args.downloads / SITES["Total"][0], index_col=0,
                          usecols=["timestamp", "Global_Horizontal_Radiation", "Weather_Temperature_Celsius"]
                          ).rename(columns=RAW_DKASC_COLUMNS)
    weather.index = pd.to_datetime(weather.index)
    nameplate = {k: kw for k, (_, kw) in SITES.items()}
    print(f"period {P.index.min()} -> {P.index.max()}, rows {len(P):,}")

    # 1. consistency
    day = weather["GHI"].reindex(P.index) > 200
    sails_sum = P[["Sails1", "Sails2", "Sails3"]].sum(axis=1, min_count=3)
    print("\n[1] consistency (daytime, GHI > 200 W/m2)")
    print(f"   Sails1+2+3 vs SailsCombined: median |diff| {(sails_sum - P['SailsCombined'])[day].abs().median():.2f} kW, "
          f"corr {sails_sum[day].corr(P['SailsCombined'][day]):.4f}")
    parts = P[["DesertGardens"] + REFERENCE].sum(axis=1, min_count=5)
    gap = (P["Total"] - parts)[day]
    print(f"   Total - sum of 5 sites: median {gap.median():.2f} kW, |gap| > 10 kW in {(gap.abs() > 10).mean():.2%} "
          f"of rows, corr {parts[day].corr(P['Total'][day]):.4f}")

    # 2. which series did the paper use?
    if args.paper_series:
        old = read_power(args.paper_series)
        j = P.join(old.rename("paper"), how="inner").dropna()
        jd = j[weather["GHI"].reindex(j.index) > 200]
        print(f"\n[2] paper series vs sites ({len(jd):,} daytime rows, {j.index.min().date()} -> {j.index.max().date()})")
        for k in SITES:
            d = (jd["paper"] - jd[k]).abs()
            print(f"   {k:17s} corr {jd['paper'].corr(jd[k]):.4f} | median |diff| {d.median():8.2f} kW | "
                  f"identical (<0.01 kW) {(d < 0.01).mean():.3f}")
        cols = ["DesertGardens"] + REFERENCE
        A, b = jd[cols].to_numpy(), jd["paper"].to_numpy()
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        print("   least-squares weights (paper ~ sum w_i * site_i):", dict(zip(cols, np.round(coef, 3))))

    # 3. curtailment per site
    print("\n[3] apparent curtailment per site (clear-sky instants, tilt 20 deg N)")
    rows = []
    for k in SITES:
        df = weather.join(P[k].rename("Active_Power"), how="inner").dropna(subset=["Active_Power", "GHI"])
        df.index = df.index.tz_localize(TZ)
        c, scale, n = analyse(df, 20.0)
        noon = c["curtailed"][(c.index.hour >= 11) & (c.index.hour <= 13)].mean()
        p999 = P[k].quantile(0.999)
        rows.append({"site": k, "nameplate_kW": nameplate[k], "p99.9_kW": round(p999, 1),
                     "p99.9_over_nameplate": round(p999 / nameplate[k], 2), "clear_instants": n,
                     "share_curtailed": round(c["curtailed"].mean(), 3), "share_curtailed_11_13h": round(noon, 3),
                     "median_ratio": round(c["r"].median(), 2)})
        print(f"   {k:17s} p99.9/nameplate {p999 / nameplate[k]:.2f} | curtailed {c['curtailed'].mean():.1%} "
              f"| 11-13h {noon:.1%} | median r {c['r'].median():.2f}")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_dir / "sites_summary.csv", index=False)

    # geometry-light diagnostic: each site relative to the other (reference) sites, on clear
    # instants, by hour. An uncontrolled site keeps a flat ratio over the day; a curtailed
    # site shows a midday dip. Ratios are normalised by their 08-09 h value.
    ref = P[REFERENCE].sum(axis=1, min_count=4)
    ghi = weather["GHI"].reindex(P.index)
    clear_day = ghi > 400
    print("\n[4] site / reference-sites ratio by hour on high-irradiance instants (normalised to 08-09 h)")
    table = {}
    for k in ["DesertGardens"] + REFERENCE:
        others = P[[c for c in REFERENCE if c != k]].sum(axis=1, min_count=3) if k in REFERENCE else ref
        ratio = (P[k] / others)[clear_day & (others > 20)]
        by_h = ratio.groupby(ratio.index.hour).median()
        table[k] = (by_h / by_h.loc[8:9].mean()).round(2)
    table = pd.DataFrame(table).loc[8:16]
    print(table.to_string())
    if args.out_dir:
        table.to_csv(args.out_dir / "site_to_reference_ratio_by_hour.csv")

    # clear winter day, hourly profile per site
    d = "2023-06-04"
    prof = P.loc[d].resample("1h").mean()
    prof["GHI"] = weather["GHI"].loc[d].resample("1h").mean()
    print(f"\nhourly profile on clear day {d} (kW, GHI W/m2):")
    print(prof.loc[f"{d} 07:00":f"{d} 18:00"].round(0).to_string())


if __name__ == "__main__":
    main()
