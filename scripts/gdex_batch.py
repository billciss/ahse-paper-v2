"""
gdex_batch.py
=============
Drives monthly NSF NCAR GDEX subset requests of archived GFS 0.25 deg forecasts (d084001)
for one site and one parameter, in netCDF over a 0.5 deg box (multi-year requests fail on
the server; single-point CSV output loses the run and forecast-hour identity and returned
only fill values). Resumable: the state is kept in <out-dir>/state_<site>_<param>.json.

For every month: submit (at most --max-active requests open), wait until Completed,
download the tar with retries, read each file's run time, forecast hour and averaging
window, interpolate bilinearly to the site, and append rows to
<out-dir>/gfs_<site>_<param>.csv with columns
    init, fhour, win_start, win_end, value
(window-start averages; src.data_engine.nwp_features.three_hour_means() gives 3-hour means).

Usage (WSL):
    python scripts/gdex_batch.py --site yulara --param DSWRF --start 2016-04 --end 2020-05
"""

import argparse
import io
import json
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdex_gfs_request import PARAMS, SITES, call, control, token  # noqa: E402

PAD = 0.25
POLL_SECONDS = 120
DESCRIPTION = {"DSWRF": "Downward shortwave radiation flux", "T CDC": "Total cloud cover"}


def find_request(requests, param, start):
    """Latest non-failed request whose own description matches this parameter and start date.

    Requests are identified by content, never by arrival order, so that several drivers can
    run concurrently without mixing parameters.
    """
    first = pd.Timestamp(start).strftime("%Y-%m-%d %H:%M")
    hits = [r for r in requests
            if DESCRIPTION[param] in ((r.get("subset_info") or {}).get("note") or "")
            and f"Start date:  {first}" in ((r.get("subset_info") or {}).get("note") or "")
            and "netCDF" in ((r.get("subset_info") or {}).get("note") or "")
            and r.get("status") != "Error"]
    return max(hits, key=lambda r: int(r["request_index"])) if hits else None


def month_ranges(start, end):
    for m in pd.period_range(start, end, freq="M"):
        first = m.start_time
        last = m.end_time.floor("6h")
        yield str(m), first.strftime("%Y%m%d%H%M"), last.strftime("%Y%m%d%H%M")


def download(url, tok, retries=6):
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}{urllib.parse.urlencode({'token': tok})}"
    for k in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=600) as r:
                return r.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError):   # URL never printed
            time.sleep(20 * (k + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {Path(urllib.parse.urlparse(url).path).name}")


def extract(tar_bytes, lat, lon):
    import xarray as xr
    rows = []
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".nc")]
        tf.extractall(tmp, members=members, filter="data")
        for m in members:
            ds = xr.open_dataset(Path(tmp) / m.name, decode_timedelta=True).load()
            var = next(v for v in ds.data_vars if v not in ("time_bnds", "valid_date_time_range",
                                                            "ref_date_time", "forecast_hour"))
            # GFS grids use longitudes in [0, 360): a western site (e.g. NIST, -77.2) must be
            # shifted, otherwise the interpolation falls outside the grid and returns NaN
            x = lon + 360.0 if (lon < 0 and float(ds["lon"].min()) >= 0) else lon
            val = float(ds[var].isel(time=0).interp(lat=lat, lon=x).values)
            b = ds["time_bnds"].values[0]
            init = pd.to_datetime(ds["ref_date_time"].values[0].decode(), format="%Y%m%d%H")
            fh = int(ds["forecast_hour"].values[0] / np.timedelta64(1, "h"))
            rows.append((init, fh, pd.Timestamp(b[0]), pd.Timestamp(b[1]), val))
    return pd.DataFrame(rows, columns=["init", "fhour", "win_start", "win_end", "value"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=sorted(SITES), required=True)
    ap.add_argument("--param", choices=sorted(PARAMS), required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--max-active", type=int, default=3)
    ap.add_argument("--purge-after-download", action="store_true",
                    help="purge each request on GDEX once its data are saved locally")
    ap.add_argument("--out-dir", type=Path, default=Path("data_external/gfs"))
    ap.add_argument("--token-file", type=Path, default=Path.home() / ".gdex_token")
    args = ap.parse_args()

    tok = token(args.token_file)
    lat, lon = SITES[args.site]
    tag = f"{args.site}_{args.param.replace(' ', '')}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path, out_csv = args.out_dir / f"state_{tag}.json", args.out_dir / f"gfs_{tag}.csv"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    months = list(month_ranges(args.start, args.end))
    save = lambda: state_path.write_text(json.dumps(state, indent=1))  # noqa: E731

    while any(state.get(m, {}).get("status") != "done" for m, _, _ in months):
        # submit pending months while fewer than max_active are open
        open_n = sum(1 for s in state.values() if s.get("status") == "submitted")
        for m, a, b in months:
            if open_n >= args.max_active:
                break
            if state.get(m, {}).get("status") in (None, "error"):
                existing = find_request(call("status", tok)["data"], args.param, a)                     if state.get(m, {}).get("status") is None else None
                if existing is None:
                    res = call("submit", tok, control(args.site, a, b, args.param, PAD, "netCDF"))
                    if res.get("error_messages"):
                        print(m, "submit error:", res["error_messages"], flush=True)
                        if any("open requests" in e for e in res["error_messages"]):
                            break                    # account quota reached: wait for slots
                        state[m] = {"status": "error", "tries": state.get(m, {}).get("tries", 0) + 1}
                        continue
                    time.sleep(5)
                    existing = find_request(call("status", tok)["data"], args.param, a)
                state[m] = {"status": "submitted",
                            "request": existing["request_index"] if existing else None,
                            "tries": state.get(m, {}).get("tries", 0) + 1}
                open_n += 1
                print(m, "submitted", state[m]["request"], flush=True)
                save()
        # check open requests
        listing = call("status", tok)["data"]
        statuses = {r["request_index"]: r["status"] for r in listing}
        month_start = {m: a for m, a, _ in months}
        for m, s in state.items():
            if s.get("status") == "submitted" and not s.get("request") and m in month_start:
                # submission whose request id was not found right after submitting: look it up
                # again, otherwise queue the month for a new submission (it would never finish)
                found = find_request(listing, args.param, month_start[m])
                if found is not None:
                    s["request"] = found["request_index"]
                else:
                    s["status"] = None
                    print(m, "submitted request not found - queued again", flush=True)
                save()
            if s.get("status") != "submitted" or not s.get("request"):
                continue
            st = statuses.get(s["request"])
            if st == "Error":
                s["status"] = "error" if s["tries"] < 3 else "failed"
                print(m, "request error", s["request"], flush=True)
                if args.purge_after_download:
                    # an errored request holds no data, but while it exists GDEX silently
                    # refuses an identical new submission (the month would never be retried)
                    call(f"purge/{s['request']}", tok, method="DELETE")
                    print(m, "purged errored request", s["request"], flush=True)
            elif st == "Completed":
                files = call(f"get_req_files/{s['request']}", tok)["data"]["web_files"]
                # every listed netCDF file points (web_path) to the same tar archive
                tars = sorted({f["web_path"] for f in files if f["web_path"].endswith(".tar")})
                if not tars:
                    print(m, "no tar archive in the request output", flush=True)
                    s["status"] = "failed"
                    save()
                    continue
                raw_dir = args.out_dir / "raw" / tag
                raw_dir.mkdir(parents=True, exist_ok=True)
                frames = []
                try:
                    for k, u in enumerate(tars):
                        blob = download(u, tok)
                        (raw_dir / f"{m}_{k}.tar").write_bytes(blob)      # kept for re-extraction
                        frames.append(extract(blob, lat, lon))
                except RuntimeError as e:            # transient download failure: retry next poll
                    print(m, "download failed, retried at the next poll:", e, flush=True)
                    continue
                df = pd.concat(frames, ignore_index=True)
                if df["value"].isna().all():         # never append an empty month
                    print(m, "all extracted values are NaN - not appended", flush=True)
                    s["status"] = "failed"
                    save()
                    continue
                months_found = sorted(df["init"].dt.strftime("%Y-%m").unique())
                if months_found != [m]:              # never append data of another month
                    print(m, "archive covers", months_found, "- not appended", flush=True)
                    s["status"] = "failed"
                    save()
                    continue
                df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
                s["status"], s["rows"] = "done", len(df)
                print(m, "done", len(df), "rows", flush=True)
                if args.purge_after_download:        # frees one of the 10 request slots
                    call(f"purge/{s['request']}", tok, method="DELETE")
                    print(m, "purged request", s["request"], flush=True)
            save()
        if all(state.get(m, {}).get("status") in ("done", "failed") for m, _, _ in months):
            break
        time.sleep(POLL_SECONDS)
    failed = [m for m, s in state.items() if s.get("status") == "failed"]
    print("finished; failed months:", failed or "none")


if __name__ == "__main__":
    main()
