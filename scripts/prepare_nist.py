"""
prepare_nist.py
===============
Builds the NIST Gaithersburg cross-site dataset for the AHSE pipeline.

Source  : NIST Campus Photovoltaic (PV) Arrays and Weather Station Data Sets
          Healy (2020), doi:10.18434/M3S67G ; Boyd et al., J. Res. NIST 122:040 (2017)
          Bulk download: https://pvdata.nist.gov  ->  Downloads "1-min avg." -> Bulk Download
Site    : Gaithersburg, Maryland, USA  (Koppen Cfa, humid subtropical)
Target  : MEASURED inverter AC power (InvPAC_kW_Avg) of one array
Weather : NIST weather station WS_1 (OneMin table)
          GHI  <- RefCell1_Wm2_Avg (horizontal, factory-calibrated Si reference cell, W/m2)
          The thermopile channels (Pyr*_mV) are raw mV and are NOT used.
Time    : loggers record Local Standard Time all year (no DST) -> fixed UTC-05:00

The output is a gap-free 24-h grid (nights included),
so that L = H = 96 steps really spans 24 h at 15 min (24 steps at 1 h).

Expected input layout (any nesting, CSV / CSV.GZ / ZIP accepted, Campbell TOA5 headers handled):
    <raw_dir>/<array>/...   e.g. data_external/nist/Ground/*.csv
    <raw_dir>/WS_1/...

Usage (WSL):
    python scripts/prepare_nist.py --raw-dir data_external/nist --array Ground \
        --start 2016-01-01 --end 2018-12-31
Outputs:
    data/processed/nist_15min.csv, data/processed/nist_1h.csv
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

BASE = Path(__file__).resolve().parent.parent
TZ = "Etc/GMT+5"  # POSIX sign convention: Etc/GMT+5 == UTC-05:00 (EST, no DST)

# Coordinates and geometry from the dataset metadata (Boyd et al. 2017).
SITES = {
    "Ground": dict(lat=39.1319, lon=-77.2141, alt=138, kw_dc=271.0),
    "Roof":   dict(lat=39.1354, lon=-77.2156, alt=149, kw_dc=73.0),
    "Canopy": dict(lat=39.1385, lon=-77.2155, alt=137, kw_dc=243.0),
}
WS = dict(lat=39.1374, lon=-77.2187, alt=158)

POWER_COLS = ["InvPAC_kW_Avg", "PwrMtrP_kW_Avg"]  # first available wins
WEATHER_MAP = {
    "RefCell1_Wm2_Avg": "GHI",
    "AirTemp_C_Avg": "Temperature",
    "AirPres_kPa_Avg": "Pressure",   # converted to hPa below (Yulara units)
    "WindSpeedAve_ms": "Wind_Speed",
    "WindDirAve_deg": "Wind_Dir",
}
# Fallback when WS_1 files are unavailable: the array's own sensors. Identified on 2016 clear
# days: Pyra1 follows the clear-sky GHI (horizontal thermopile, W/m2); Pyra2, RefCell1 and
# SEWSPOAIrrad are plane-of-array. The arrays have no barometer: pressure is set to the standard
# pressure of the site altitude (a constant, i.e. no information for the models).
ARRAY_WEATHER_MAP = {
    "Pyra1_Wm2_Avg": "GHI",
    "AmbTemp_C_Avg": "Temperature",
    "WindSpeedAve_ms": "Wind_Speed",
    "WindDirAve_deg": "Wind_Dir",
}
TIME_CANDIDATES = ["TIMESTAMP", "timestamp", "Timestamp", "index", "datetime", "time"]

# Short gaps are interpolated here; longer ones are reported and must be handled
# (the pipeline would otherwise interpolate them linearly across days).
MAX_INTERP_MIN = 60
OUT_COLUMNS = [
    "Active_Power", "Wind_Speed", "Temperature", "GHI", "Wind_Dir", "Pressure",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos", "day_of_week",
    "zenith_angle", "clear_sky_ghi", "clear_sky_power", "kt",
]


def _read_one(buf, name):
    head = buf.read(200)
    buf.seek(0)
    is_toa5 = head.lstrip(b'"').startswith(b"TOA5")
    df = pd.read_csv(buf, skiprows=[0, 2, 3] if is_toa5 else None, low_memory=False,
                     compression="gzip" if name.endswith(".gz") else None)
    tcol = next((c for c in TIME_CANDIDATES if c in df.columns), df.columns[0])
    df.index = pd.to_datetime(df.pop(tcol), errors="coerce")
    if df.index.tz is not None:  # keep everything in naive Local Standard Time
        df.index = df.index.tz_convert(TZ).tz_localize(None)
    return df[df.index.notna()]


def read_folder(folder: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                for n in z.namelist():
                    if n.lower().endswith((".csv", ".csv.gz", ".dat")):
                        frames.append(_read_one(io.BytesIO(z.read(n)), n.lower()))
        elif f.name.lower().endswith((".csv", ".csv.gz", ".dat")):
            with open(f, "rb") as fh:
                frames.append(_read_one(io.BytesIO(fh.read()), f.name.lower()))
    if not frames:
        sys.exit(f"No CSV/ZIP files found under {folder}")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"  {folder.name}: {len(df):,} rows, {df.index.min()} -> {df.index.max()}")
    return df.apply(pd.to_numeric, errors="coerce")


def report_gaps(s: pd.Series, label: str, freq: str):
    missing = s.isna()
    if not missing.any():
        print(f"  {label}: no gaps")
        return []
    runs = (missing != missing.shift()).cumsum()[missing]
    gaps = [(g.index[0], g.index[-1], len(g)) for _, g in s[missing].groupby(runs)]
    step_min = pd.Timedelta(freq).total_seconds() / 60
    long_gaps = [g for g in gaps if g[2] * step_min > MAX_INTERP_MIN]
    print(f"  {label}: {missing.sum():,} missing steps in {len(gaps)} gaps, "
          f"{len(long_gaps)} longer than {MAX_INTERP_MIN} min")
    for a, b, n in long_gaps[:15]:
        print(f"      {a} -> {b}  ({n * step_min / 60:.1f} h)")
    return long_gaps


def build(raw: pd.DataFrame, freq: str, site: dict, tz: str = TZ) -> pd.DataFrame:
    df = raw.resample(freq).mean()
    limit = max(1, int(MAX_INTERP_MIN / (pd.Timedelta(freq).total_seconds() / 60)))

    loc = pvlib.location.Location(site["lat"], site["lon"], tz=tz, altitude=site["alt"])
    idx = df.index.tz_localize(tz)
    sp = loc.get_solarposition(idx)
    night = (sp["apparent_zenith"] >= 90).to_numpy()

    # Night power is structurally zero; inverter tare can read slightly negative.
    df.loc[night, "Active_Power"] = 0.0
    df["Active_Power"] = df["Active_Power"].clip(lower=0)
    df.loc[night & df["GHI"].isna().to_numpy(), "GHI"] = 0.0
    df["GHI"] = df["GHI"].clip(lower=0)

    long_gaps = {c: report_gaps(df[c], c, freq) for c in ["Active_Power", "GHI", "Temperature"]}
    # pandas' `limit` would still fill the first `limit` steps of a long gap: fill only
    # gaps whose total length is <= limit, leave long ones entirely NaN.
    na = df.isna()
    run_len = na.apply(lambda s: s.groupby((~s).cumsum()).transform("sum")).where(na, 0)
    df = df.interpolate(method="time", limit_area="inside").mask(run_len > limit)

    df.index = idx
    doy, hour = df.index.dayofyear, df.index.hour + df.index.minute / 60
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365)
    df["day_of_week"] = df.index.dayofweek / 6
    df["zenith_angle"] = sp["apparent_zenith"].to_numpy()
    df["clear_sky_ghi"] = loc.get_clearsky(idx, model="ineichen")["ghi"].to_numpy()
    # clear_sky_power and kt are recomputed by the pipeline (pv_physics.py); placeholders keep
    # the column layout identical to the Yulara processed files.
    df["clear_sky_power"] = 0.0
    df["kt"] = 0.0
    return df[OUT_COLUMNS], long_gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(BASE / "data_external" / "nist"))
    ap.add_argument("--array", default="Ground", choices=sorted(SITES))
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2018-12-31")
    ap.add_argument("--out-dir", default=str(BASE / "data" / "processed"))
    ap.add_argument("--weather-source", choices=["ws1", "array"], default="ws1",
                    help="weather from station WS_1 (default) or from the array's own sensors")
    ap.add_argument("--tag", default="nist", help="output file prefix")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    print(f"Reading raw 1-min files from {raw_dir} ...")
    arr = read_folder(raw_dir / args.array)
    ws = read_folder(raw_dir / "WS_1") if args.weather_source == "ws1" else arr
    wmap = WEATHER_MAP if args.weather_source == "ws1" else ARRAY_WEATHER_MAP

    pcol = next((c for c in POWER_COLS if c in arr.columns), None)
    if pcol is None:
        sys.exit(f"None of {POWER_COLS} found in {args.array} files. Columns: {list(arr.columns)[:40]}")
    missing_w = [c for c in wmap if c not in ws.columns]
    if missing_w:
        sys.exit(f"Missing weather columns in the {args.weather_source} files: {missing_w}")

    raw = pd.concat([arr[[pcol]].rename(columns={pcol: "Active_Power"}),
                     ws[list(wmap)].rename(columns=wmap)], axis=1)
    if args.weather_source == "ws1":
        raw["Pressure"] = raw["Pressure"] * 10.0  # kPa -> hPa
    else:
        raw["Pressure"] = pvlib.atmosphere.alt2pres(SITES[args.array]["alt"]) / 100.0  # hPa, constant
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    raw = raw.loc[args.start:end]
    # Enforce a complete 1-min grid so resampling never skips a period.
    raw = raw.reindex(pd.date_range(args.start, end, freq="1min"))
    print(f"Power column: {pcol}  |  period {args.start} -> {args.end}  |  array {args.array}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    site = SITES[args.array]
    any_long = False
    for freq, tag in [("15min", "15min"), ("1h", "1h")]:
        print(f"\n[{tag}]")
        df, long_gaps = build(raw, freq, site)
        any_long |= any(long_gaps.values())
        per_day = df.groupby(df.index.date).size()
        assert (per_day == per_day.mode()[0]).all(), "grid is not complete: unequal rows per day"
        path = out_dir / f"{args.tag}_{tag}.csv"
        df.to_csv(path, index_label="timestamp")
        print(f"  -> {path}  ({len(df):,} rows, {per_day.mode()[0]} rows/day, "
              f"{df.isna().any(axis=1).sum():,} rows still NaN)")
        print(f"     P_ac max {df['Active_Power'].max():.1f} kW (array DC {site['kw_dc']} kW), "
              f"GHI max {df['GHI'].max():.0f} W/m2")

    if any_long:
        print("\nWARNING: gaps longer than 60 min remain as NaN. The AHSE pipeline would interpolate "
              "them linearly; shorten --start/--end to avoid them or document them before training.")


if __name__ == "__main__":
    main()
