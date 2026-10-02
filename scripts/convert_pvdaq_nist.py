"""
convert_pvdaq_nist.py
=====================
Converts PVDAQ system 4902 (NIST_Ground_1, the NIST Ground array; public OEDI data lake,
doi:10.25984/1846021) daily parquet files to the wide 1-min CSV layout read by
scripts/prepare_nist.py (--weather-source array).

PVDAQ stores the values of the NIST loggers unchanged (checked against the NIST archive
onemin-Ground-2016.zip: AC power, reference cell and ambient temperature identical) except the
horizontal pyranometer, published in mV (Pyra1_mV_Avg) whereas NIST publishes W/m2.

The conversion is not constant (checked 2026-10-02):
  - fitted month by month against the NIST 2016 archive, NIST used W/m2 = 106.597 mV - 6.396
    until May 2016 and 118.804 mV - 7.128 afterwards (residuals ~1e-4 W/m2 in each regime);
  - the raw mV per unit of reference-cell irradiance (whose calibration is stable: AC power per
    reference-cell irradiance is the same in 2016 and 2017) drops by ~16% between 2017-10-10 and
    2017-10-17, most likely a sensor change whose new constant is unknown.
Hence: 2016 uses the W/m2 values published by NIST (the archive), 2017 uses the post-May-2016
constant, and the output stops at --end (default 2017-10-10).

Usage (WSL):
    python scripts/convert_pvdaq_nist.py --pvdaq data_external/pvdaq_4902 \
        --nist-zip data_external/nist/Ground/onemin-Ground-2016.zip --out data_external/nist_pvdaq/Ground
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {82607: "InvPAC_kW_Avg", 82593: "Pyra1_mV_Avg", 82595: "RefCell1_Wm2_Avg",
           82596: "AmbTemp_C_Avg", 82665: "WindSpeedAve_ms", 82666: "WindDirAve_deg"}
TZ = "Etc/GMT+5"   # loggers record Local Standard Time (UTC-05:00) all year


def read_month(folder: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(f, columns=["measured_on", "metric_id", "value"]) for f in sorted(folder.glob("*.parquet"))]
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts)
    d = d[d.metric_id.isin(METRICS)]
    w = d.pivot_table(index="measured_on", columns="metric_id", values="value", aggfunc="mean")
    w = w.rename(columns=METRICS)
    w.index = pd.DatetimeIndex(w.index)
    return w


def calibration(nist_zip: Path, pvdaq: Path):
    z = zipfile.ZipFile(nist_zip)
    rows = []
    for month in range(1, 13):
        w = read_month(pvdaq / "2016" / f"{month:02d}")
        if w.empty:
            continue
        frames = []
        for n in z.namelist():
            if f"2016/{month:02d}/" in n and n.endswith(".csv"):
                d = pd.read_csv(io.BytesIO(z.read(n)), usecols=["TIMESTAMP", "Pyra1_Wm2_Avg"])
                frames.append(d)
        n = pd.concat(frames)
        n.index = pd.DatetimeIndex(pd.to_datetime(n.pop("TIMESTAMP"), utc=True)).tz_convert(TZ).tz_localize(None)
        j = w[["Pyra1_mV_Avg"]].join(n, how="inner").dropna()
        j = j[j.Pyra1_Wm2_Avg > 20]                      # daytime
        a, b = np.polyfit(j.Pyra1_mV_Avg, j.Pyra1_Wm2_Avg, 1)
        resid = j.Pyra1_Wm2_Avg - (a * j.Pyra1_mV_Avg + b)
        rows.append({"month": month, "n": len(j), "slope": a, "offset": b,
                     "max_abs_residual": float(resid.abs().max())})
    cal = pd.DataFrame(rows)
    return cal, float(cal.slope.median()), float(cal.offset.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvdaq", type=Path, required=True)
    ap.add_argument("--nist-zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--end", default="2017-10-10", help="last day with a known pyranometer constant")
    args = ap.parse_args()

    cal, _, _ = calibration(args.nist_zip, args.pvdaq)
    print(cal.round(4).to_string(index=False))
    a, b = (float(v) for v in cal[cal.month >= 6][["slope", "offset"]].median())   # post-May-2016 constant
    print(f"after 2016: W/m2 = {a:.4f} * mV + {b:.3f}; 2016: NIST published values")
    z = zipfile.ZipFile(args.nist_zip)
    nist = pd.concat([pd.read_csv(io.BytesIO(z.read(n)), usecols=["TIMESTAMP", "Pyra1_Wm2_Avg"])
                      for n in z.namelist() if n.endswith(".csv")])
    nist.index = pd.DatetimeIndex(pd.to_datetime(nist.pop("TIMESTAMP"), utc=True)).tz_convert(TZ).tz_localize(None)
    nist = nist[~nist.index.duplicated()]["Pyra1_Wm2_Avg"]
    last = pd.Timestamp(args.end) + pd.Timedelta(days=1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "pyra1_calibration.json").write_text(json.dumps(
        {"post_may_2016": {"slope": a, "offset": b}, "end": args.end,
         "monthly_2016": cal.to_dict(orient="records")}, indent=1))

    for year_dir in sorted(p for p in args.pvdaq.iterdir() if p.is_dir() and p.name.isdigit()):
        for month_dir in sorted(year_dir.iterdir()):
            w = read_month(month_dir)
            if w.empty:
                continue
            w["Pyra1_Wm2_Avg"] = a * w.pop("Pyra1_mV_Avg") + b
            in_2016 = w.index.year == 2016
            w.loc[in_2016, "Pyra1_Wm2_Avg"] = nist.reindex(w.index[in_2016]).to_numpy()
            w = w[w.index < last]
            if w.empty:
                continue
            w.index = w.index.tz_localize(TZ)
            out = args.out / f"pvdaq4902_{year_dir.name}-{month_dir.name}.csv.gz"
            w.to_csv(out, index_label="TIMESTAMP")
            print(f"{year_dir.name}-{month_dir.name}: {len(w):,} rows", flush=True)


if __name__ == "__main__":
    main()
