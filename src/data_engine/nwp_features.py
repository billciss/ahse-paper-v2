"""
NWP (GFS) covariates for day-ahead forecasting without look-ahead.

For a forecast issued at t0 (label of the last observation) for target steps t0+1 .. t0+H,
only the latest GFS run that was PUBLISHED by t0 may be used:
    usable run = max{ init : init + availability_delay <= t0 }
Targets are interval means labelled by the START of the interval (pandas resample default),
so target tau covers [tau, tau + step). It takes, from that run, the 3-hour mean of the
window (valid - 3 h, valid] that contains the interval midpoint tau + step/2.

Input: a long table with columns init, valid (UTC, window end), and one column per variable
holding 3-hour means (three_hour_means() converts the GFS window-start averages written by
scripts/gdex_batch.py; load_gfs() builds the table from those CSV files).
"""

from typing import Sequence

import numpy as np
import pandas as pd
import pvlib

GFS_AVAILABILITY_DELAY = pd.Timedelta(hours=5)


def three_hour_means(df: pd.DataFrame) -> pd.DataFrame:
    """3-hour means over (valid - 3 h, valid], valid = win_end, from window-start averages.

    GFS stores averages from the start of the 6-hour cycle: the 3-hour average for
    (f-3, f] when f is the first half of the cycle, the 6-hour average (f-6, f] otherwise,
    so mean(f-3, f] = 2 * avg6(f) - avg3(f-3).
    """
    df = df.sort_values(["init", "fhour"]).drop_duplicates(["init", "fhour"]).copy()
    df["hours"] = (df["win_end"] - df["win_start"]) / pd.Timedelta("1h")
    out = []
    for _, run in df.groupby("init"):
        run = run.set_index("win_end")
        m3 = run["value"].copy()
        six = (run["hours"] == 6).to_numpy()
        prev = run["value"].reindex(run.index - pd.Timedelta("3h")).to_numpy()
        m3[six] = 2 * run["value"].to_numpy()[six] - prev[six]
        out.append(pd.DataFrame({"init": run["init"].to_numpy(), "valid": run.index,
                                 "value_3h": m3.clip(lower=0).to_numpy()}))   # de-accumulation noise
    return pd.concat(out, ignore_index=True)


def load_gfs(dswrf_csv, tcdc_csv) -> pd.DataFrame:
    """Table init, valid (UTC), dswrf (W/m2), tcdc (%) of 3-hour means, from gdex_batch CSVs."""
    parts = []
    for path, name in [(dswrf_csv, "dswrf"), (tcdc_csv, "tcdc")]:
        raw = pd.read_csv(path, parse_dates=["init", "win_start", "win_end"])
        parts.append(three_hour_means(raw).rename(columns={"value_3h": name}).set_index(["init", "valid"]))
    g = parts[0].join(parts[1], how="outer").reset_index()
    g["init"] = pd.to_datetime(g["init"]).dt.tz_localize("UTC")
    g["valid"] = pd.to_datetime(g["valid"]).dt.tz_localize("UTC")
    return g


def gfs_target_features(gfs: pd.DataFrame, time_index: pd.DatetimeIndex, target_start_rows: np.ndarray,
                        horizon_steps: int, step: pd.Timedelta, location: pvlib.location.Location) -> dict:
    """Per-target GFS covariates for forecasts whose first target is time_index[start].

    Returns arrays of shape (n, horizon_steps): dswrf, tcdc (latest run published at the
    issue time = time of the last observation), cs (clear-sky GHI at the target time) and
    kc (GFS clearness index: 3-hour GFS radiation over the clear-sky mean of the SAME 3-hour
    window; NaN where the run has no value, 0 at night).
    """
    tidx = pd.DatetimeIndex(time_index)
    starts = np.asarray(target_start_rows)
    n = len(starts)
    issue = tidx[starts - 1].tz_convert("UTC")
    cov = nwp_covariates(gfs, issue, horizon_steps, step, ["dswrf", "tcdc"])
    targets = tidx[(starts[:, None] + np.arange(horizon_steps)[None, :]).ravel()]
    cs = location.get_clearsky(targets)["ghi"].to_numpy().reshape(n, horizon_steps)
    t_utc = targets.tz_convert("UTC")
    t5 = pd.date_range(t_utc.min().floor("D"), t_utc.max().ceil("D") + pd.Timedelta(days=1), freq="5min")
    cs_3h = location.get_clearsky(t5)["ghi"].resample("3h", label="right", closed="right").mean()
    valid = (t_utc + step / 2).ceil("3h")          # same window as nwp_covariates (inits at 0/6/12/18 UTC)
    cs_win = cs_3h.reindex(valid).to_numpy().reshape(n, horizon_steps)
    kc = np.clip(cov[:, :, 0] / np.maximum(cs_win, 20.0), 0, 1.5)
    kc[cs_win < 20] = 0.0
    return {"dswrf": cov[:, :, 0], "tcdc": cov[:, :, 1], "cs": cs, "kc": kc}


def nwp_covariates(nwp: pd.DataFrame, issue_times: Sequence[pd.Timestamp], horizon_steps: int,
                   step: pd.Timedelta, variables: Sequence[str],
                   availability_delay: pd.Timedelta = GFS_AVAILABILITY_DELAY) -> np.ndarray:
    """Array (n_issues, horizon_steps, n_vars); NaN where the chosen run has no value."""
    nwp = nwp.copy()
    nwp["init"] = pd.to_datetime(nwp["init"], utc=True)
    nwp["valid"] = pd.to_datetime(nwp["valid"], utc=True)
    table = nwp.set_index(["init", "valid"])[list(variables)].sort_index()
    runs = pd.DatetimeIndex(nwp["init"].unique()).sort_values()
    published = runs + availability_delay

    issue = pd.DatetimeIndex(pd.to_datetime(issue_times, utc=True))
    out = np.full((len(issue), horizon_steps, len(variables)), np.nan)
    k = published.searchsorted(issue, side="right") - 1              # latest published run
    three_h = pd.Timedelta(hours=3)
    for i, (t0, ki) in enumerate(zip(issue, k)):
        if ki < 0:
            continue
        init = runs[ki]
        mids = pd.date_range(t0 + step, periods=horizon_steps, freq=step) + step / 2
        # first 3-hour boundary >= interval midpoint, counted from the run's initial time
        valid = init + pd.to_timedelta(3 * np.ceil((mids - init) / three_h), unit="h")
        try:
            vals = table.loc[init].reindex(valid).to_numpy()
        except KeyError:
            continue
        out[i] = vals
    return out
