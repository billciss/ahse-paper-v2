"""
gdex_gfs_request.py
===================
Submits, monitors and downloads NSF NCAR GDEX subset requests for archived GFS 0.25 deg
forecasts (dataset d084001) at a single grid point, as CSV.

The API token is read from a file created by the user (default ~/.gdex_token, the token
shown on https://gdex.ucar.edu/accounts/profile). The token is never printed.

Requested fields (all runs 00/06/12/18 UTC, so that the latest run available at each
forecast issue time can be selected later):
  DSWRF  downward shortwave radiation flux, surface (SFC)     -> one request
  T CDC  total cloud cover, entire atmosphere (EATM)           -> one request
  products: 3-hour and 6-hour averages from initial+0 up to initial+48

Usage (WSL):
    python scripts/gdex_gfs_request.py submit --site yulara --start 201604010000 --end 202005150000
    python scripts/gdex_gfs_request.py status
    python scripts/gdex_gfs_request.py download --request <index> --out-dir data_external/gfs
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://gdex.ucar.edu/api"
SITES = {"yulara": (-25.24, 131.04), "nist": (39.1319, -77.2141)}
# one request per parameter: combining level types in one "level" string is rejected
PARAMS = {"DSWRF": "SFC:0", "T CDC": "EATM:0"}
PRODUCTS = []
for a in range(0, 48, 6):
    PRODUCTS += [f"3-hour Average (initial+{a} to initial+{a + 3})",
                 f"6-hour Average (initial+{a} to initial+{a + 6})"]


def token(path: Path) -> str:
    if not path.exists():
        sys.exit(f"Token file {path} not found: create it with your GDEX API token (chmod 600).")
    return path.read_text().strip()


def call(endpoint, tok, data=None, method=None):
    # trailing slash: the server redirects without it, which turns a POST into a GET
    url = f"{API}/{endpoint.rstrip('/')}/?{urllib.parse.urlencode({'token': tok})}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"},
                                 method=method)
    for attempt in range(12):                     # up to ~1 h of patience in total
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:       # never print the URL: it carries the token
            # redirects during maintenance (e.g. a transient 308), throttling and server
            # errors are retried; genuine client errors stop the run
            if not (e.code in (301, 302, 307, 308, 429) or e.code >= 500):
                sys.exit(f"GDEX API error {e.code} on '{endpoint}': {e.read().decode()[:300]}")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass                                  # transient SSL/network failure
        time.sleep(min(60 * (attempt + 1), 600))
    sys.exit(f"GDEX API unreachable on '{endpoint}' after 12 attempts")


def control(site, start, end, param, pad=0.0, oformat="csv"):
    lat, lon = SITES[site]
    return {"dataset": "d084001", "date": f"{start}/to/{end}", "datetype": "init",
            "param": param, "level": PARAMS[param], "product": "/".join(PRODUCTS),
            "oformat": oformat, "nlat": lat + pad, "slat": lat - pad, "wlon": lon - pad, "elon": lon + pad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["submit", "status", "download", "show"])
    ap.add_argument("--site", choices=sorted(SITES))
    ap.add_argument("--start", help="YYYYMMDDHHMM")
    ap.add_argument("--end", help="YYYYMMDDHHMM")
    ap.add_argument("--request", help="request index")
    ap.add_argument("--param", choices=sorted(PARAMS), help="submit only this parameter")
    ap.add_argument("--pad", type=float, default=0.0, help="half-width (deg) of the box around the site")
    ap.add_argument("--format", default="csv", choices=["csv", "netCDF"],
                    help="netCDF keeps run and forecast-hour dimensions (CSV loses them)")
    ap.add_argument("--out-dir", type=Path, default=Path("data_external/gfs"))
    ap.add_argument("--token-file", type=Path, default=Path.home() / ".gdex_token")
    args = ap.parse_args()

    if args.action == "show":                      # control file only, no network call
        for param in PARAMS:
            print(json.dumps(control(args.site, args.start, args.end, param), indent=1))
        return
    tok = token(args.token_file)
    if args.action == "submit":
        for param in ([args.param] if args.param else PARAMS):
            res = call("submit", tok, control(args.site, args.start, args.end, param, args.pad, args.format))
            print(param, json.dumps({k: res.get(k) for k in ("http_response", "messages", "error_messages", "result")}))
    elif args.action == "status":
        res = call("status/" + (args.request or ""), tok)
        for r in (res.get("result") or res.get("data") or []) if isinstance(res.get("result") or res.get("data"), list) \
                else [res.get("result") or res.get("data")]:
            print({k: r.get(k) for k in ("request_index", "status", "date_rqst", "date_ready", "date_purge")})
    elif args.action == "download":
        res = call(f"get_req_files/{args.request}", tok)
        files = (res.get("result") or res.get("data") or {}).get("web_files", [])
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            url = f["web_path"] if isinstance(f, dict) else f
            dest = args.out_dir / Path(urllib.parse.urlparse(url).path).name
            sep = "&" if "?" in url else "?"
            urllib.request.urlretrieve(f"{url}{sep}{urllib.parse.urlencode({'token': tok})}", dest)
            print("downloaded", dest, dest.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
