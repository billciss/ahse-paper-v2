"""
quantify_curtailment.py
=======================
Quantifies apparent curtailment of the Yulara PV output (off-grid mini-grid).

Idea: on CLEAR-SKY instants (detected from measured GHI), weather cannot explain a
large shortfall of PV power. The expected clear-sky power is c * POA_cs(tilt, azimuth)
for fixed, north-facing arrays; c is the 95th percentile of P / POA_cs over clear
instants (i.e. we assume at least 5% of clear instants are not curtailed). The ratio
r = P / (c * POA_cs) then measures delivered vs. available power; r < R_CURTAILED flags
apparent curtailment. Results are reported by hour, month, year, weekday and daily
maximum temperature, for several plausible tilts (sensitivity).

Usage:
    python scripts/quantify_curtailment.py data/raw/data_5min.csv --out-dir results/curtailment
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

LAT, LON, ALT, TZ = -25.24, 131.04, 492, "Australia/Darwin"
TILTS = [15.0, 20.0, 25.0]
AZIMUTH = 0.0                 # pvlib convention: 0 = north (arrays face the equator)
R_CURTAILED = 0.8
CLEAR_BAND = (0.9, 1.1)
CLEAR_STD = 0.05
ZENITH_MAX = 70.0


RAW_DKASC_COLUMNS = {  # names in DKASC downloads -> names used by the pipeline
    "Global_Horizontal_Radiation": "GHI",
    "Weather_Temperature_Celsius": "Temperature",
}


def load(path):
    df = pd.read_csv(path, index_col=0).rename(columns=RAW_DKASC_COLUMNS)
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
    return df


def clear_instants(df, cs, sp):
    ghi_cs = cs["ghi"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ghi_cs > 100, df["GHI"].to_numpy() / ghi_cs, np.nan)
    std = pd.Series(ratio, index=df.index).rolling("60min", min_periods=6).std().to_numpy()
    return ((ratio >= CLEAR_BAND[0]) & (ratio <= CLEAR_BAND[1]) & (std <= CLEAR_STD)
            & (sp["apparent_zenith"].to_numpy() < ZENITH_MAX))


def analyse(df, tilt):
    loc = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
    cs = loc.get_clearsky(df.index)
    sp = loc.get_solarposition(df.index)
    poa = pvlib.irradiance.get_total_irradiance(
        tilt, AZIMUTH, sp["apparent_zenith"], sp["azimuth"], cs["dni"], cs["ghi"], cs["dhi"]
    )["poa_global"].fillna(0).to_numpy()
    clear = clear_instants(df, cs, sp) & (poa > 200)
    p = df["Active_Power"].to_numpy()
    scale = np.nanpercentile(p[clear] / poa[clear], 95)
    r = np.full(len(df), np.nan)
    r[clear] = p[clear] / (scale * poa[clear])
    out = pd.DataFrame({"r": r, "curtailed": r < R_CURTAILED}, index=df.index)[clear]
    return out, scale, int(clear.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    df = load(args.csv)
    tmax = df["Temperature"].resample("1D").max()

    summary = {}
    for tilt in TILTS:
        c, scale, n = analyse(df, tilt)
        summary[tilt] = c
        print(f"\n=== tilt {tilt:.0f}° N | clear instants {n:,} | scale {scale:.3f} kW/(W/m2) "
              f"| implied clear-sky peak {scale * 1000:.0f} kW (nameplate 1820 kW)")
        print(f"share curtailed (r < {R_CURTAILED}) : {c['curtailed'].mean():.1%} | "
              f"median r {c['r'].median():.2f}")
        if tilt != 20.0:
            continue
        hour = c.index.hour
        print("by hour   :", c.groupby(hour)["curtailed"].mean().round(2).to_dict())
        print("by month  :", c.groupby(c.index.month)["curtailed"].mean().round(2).to_dict())
        print("by year   :", c.groupby(c.index.year)["curtailed"].mean().round(2).to_dict())
        print("by weekday:", c.groupby(c.index.dayofweek)["curtailed"].mean().round(2).to_dict())
        day = c.groupby(c.index.date)["curtailed"].mean()
        t = tmax.reindex(pd.to_datetime(day.index).tz_localize(TZ))
        bins = pd.cut(t.to_numpy(), [-10, 20, 25, 30, 35, 50])
        print("by daily Tmax (°C):", day.groupby(bins, observed=True).mean().round(2).to_dict())
        print("corr(daily curtailed share, daily Tmax):", round(float(np.corrcoef(day.to_numpy(), t.to_numpy())[0, 1]), 3))
        # are curtailed levels flat (set-points)? histogram of P on curtailed clear instants
        pc = df.loc[c.index[c["curtailed"]], "Active_Power"]
        print("P on curtailed clear instants, quantiles 10/25/50/75/90 (kW):",
              np.round(pc.quantile([.1, .25, .5, .75, .9]).to_numpy(), 0))
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            c.to_csv(args.out_dir / "clear_instants_ratio_tilt20.csv")
            (c.assign(month=c.index.month, hour=c.index.hour)
              .pivot_table(index="month", columns="hour", values="curtailed", aggfunc="mean")
              .round(3).to_csv(args.out_dir / "curtailed_share_month_by_hour.csv"))

    both = pd.concat({t: s["curtailed"] for t, s in summary.items()}, axis=1).dropna()
    print("\nagreement of the curtailment flag across tilts:", round(float((both.nunique(axis=1) == 1).mean()), 3))


if __name__ == "__main__":
    main()
