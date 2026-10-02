"""
prepare_yulara_sites.py
=======================
Builds AHSE-pipeline datasets (gap-free 24-h grids at 15 min and 1 h) from the DKASC
Yulara per-site downloads, for three targets:

  neighbours   : sum of the four UNCURTAILED sites (Service Station, Sails in the Desert,
                 Laundry, Connellan Airport; 699 kW) - weather-driven PV at Yulara
  dg_delivered : Desert Gardens as delivered (the series used in the original paper;
                 curtailed by the mini-grid)
  dg_available : Desert Gardens available power reconstructed by
                 censored_available_power.py (needs its output CSV)

Weather (GHI, temperature, pressure, wind) comes from the "Total of all sites" file
(same station for all sites). Default period for `neighbours`: 2016-04-02 -> 2020-05-13,
the longest window without neighbour outages (see yulara_data_quality.py).

Usage (WSL):
    python scripts/prepare_yulara_sites.py --downloads /mnt/c/Users/billc/Downloads \
        --target neighbours --out-dir data/processed
Outputs: data/processed/yulara_<target>_{15min,1h}.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_nist import OUT_COLUMNS, build  # noqa: E402
from yulara_sites_analysis import SITES, read_power  # noqa: E402

TZ = "Australia/Darwin"
SITE = dict(lat=-25.24, lon=131.04, alt=492)
NEIGHBOURS = ["ServiceStation", "SailsCombined", "Laundry", "ConnellanAirport"]
DEFAULT_PERIODS = {"neighbours": ("2016-04-02", "2020-05-13"),
                   "dg_delivered": ("2016-04-02", "2026-02-18"),
                   "dg_available": ("2016-04-02", "2026-02-18")}
WEATHER = {"Global_Horizontal_Radiation": "GHI", "Weather_Temperature_Celsius": "Temperature",
           "Air_Pressure": "Pressure", "Wind_Speed": "Wind_Speed", "Wind_Direction": "Wind_Dir"}


def load_weather(downloads: Path) -> pd.DataFrame:
    w = pd.read_csv(downloads / SITES["Total"][0], index_col=0, usecols=["timestamp", *WEATHER]).rename(columns=WEATHER)
    w.index = pd.to_datetime(w.index)
    return w[~w.index.duplicated()]


def load_target(target: str, downloads: Path, available_csv: Path) -> pd.Series:
    if target == "neighbours":
        parts = pd.DataFrame({s: read_power(downloads / SITES[s][0]) for s in NEIGHBOURS})
        return parts.sum(axis=1, min_count=len(NEIGHBOURS))
    if target == "dg_delivered":
        return read_power(downloads / SITES["DesertGardens"][0])
    if target == "dg_available":
        a = pd.read_csv(available_csv, index_col=0, usecols=["timestamp", "DG_available"])["DG_available"]
        a.index = pd.to_datetime(a.index, utc=True).tz_convert(TZ).tz_localize(None)
        # 15-min daytime-only series: upsample to 1 min by forward fill inside each 15-min slot,
        # night rows become NaN and are set to zero by build()
        return a.resample("1min").ffill(limit=14)
    raise ValueError(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path, required=True)
    ap.add_argument("--target", choices=sorted(DEFAULT_PERIODS), required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--available-csv", type=Path, default=Path("data/processed/yulara_dg_available_15min.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    start, end = args.start or DEFAULT_PERIODS[args.target][0], args.end or DEFAULT_PERIODS[args.target][1]

    raw = load_weather(args.downloads).join(load_target(args.target, args.downloads, args.available_csv)
                                           .rename("Active_Power"), how="outer")
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    raw = raw.loc[start:end_ts].reindex(pd.date_range(start, end_ts, freq="1min"))
    print(f"target {args.target}: {start} -> {end}, power max {raw['Active_Power'].max():.0f} kW")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for freq in ["15min", "1h"]:
        print(f"\n[{freq}]")
        df, long_gaps = build(raw, freq, SITE, tz=TZ)
        per_day = df.groupby(df.index.date).size()
        assert (per_day == per_day.mode()[0]).all(), "grid is not complete"
        path = args.out_dir / f"yulara_{args.target}_{freq}.csv"
        df[OUT_COLUMNS].to_csv(path, index_label="timestamp")
        print(f"  -> {path} ({len(df):,} rows, {per_day.mode()[0]} rows/day, "
              f"{df.isna().any(axis=1).sum():,} rows with NaN)")


if __name__ == "__main__":
    main()
