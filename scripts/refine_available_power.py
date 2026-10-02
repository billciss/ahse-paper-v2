"""
refine_available_power.py
=========================
Refines and quantifies the uncertainty of the Desert Gardens (DG) available-power
reconstruction (see reconstruct_available_power.py).

For each held-out year (leave-one-year-out), clear-sky envelopes are fitted on the other
years (neighbours: all rows; DG: uncurtailed pool U only), then three variants of the
clear-sky-index transfer are compared on the held-out U instants:
  kt_sum      : k_c = sum(neighbours) / sum(neighbour envelopes)          (baseline)
  kt_weighted : k_c = sum_i w_i k_c,i, w >= 0, sum w = 1, fitted on training U (NNLS)
  kt_wsmooth  : as kt_weighted with each k_c,i smoothed by a centred 3-step (45-min) mean
                (legitimate: the reconstruction is an offline target, not a forecast)
DG_available = k_c * DG envelope.

Errors are reported by output level and by sky class (neighbour k_c and its variability).
Uncertainty: split-conformal intervals (target 90%) on the relative residual
(y - y_hat) / DG envelope, Mondrian by sky class, calibrated for each held-out year on
the out-of-year residuals of the OTHER years; coverage is measured on the held-out year.
Limitation: U contains no clear-sky midday instants, so coverage there cannot be checked
directly; the midday admissibility check (delivered <= available) is reported instead.

Usage (WSL):
    python scripts/refine_available_power.py --downloads /mnt/c/Users/billc/Downloads \
        --out-csv data/processed/yulara_dg_available_15min.csv --metrics results/curtailment/refined_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_available_power import NEIGHBOURS, _envelope, build_frame, metrics  # noqa: E402

ALPHA = 0.10
SKY_CLASSES = ["clear", "variable", "overcast"]


def sky_class(kc, kc_std):
    """clear: k_c >= 0.85 and stable; overcast: k_c < 0.5; variable: the rest."""
    cls = np.full(len(kc), "variable", dtype=object)
    cls[(kc >= 0.85) & (kc_std < 0.05)] = "clear"
    cls[kc < 0.5] = "overcast"
    return cls


def conformal_q(scores, alpha=ALPHA):
    s = np.sort(scores[np.isfinite(scores)])
    n = s.size
    if n == 0:
        return np.nan
    return s[min(n, int(np.ceil((n + 1) * (1 - alpha)))) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path)
    ap.add_argument("--metrics", type=Path)
    args = ap.parse_args()

    df = build_frame(args.downloads)
    years = sorted(df.index.year.unique())
    variants = ["kt_sum", "kt_weighted", "kt_wsmooth"]
    pred = {v: pd.Series(np.nan, index=df.index) for v in variants}
    env_dg = pd.Series(np.nan, index=df.index)
    kc_sum = pd.Series(np.nan, index=df.index)
    weights = {}

    for yr in years:
        train = df[df.index.year != yr]
        envs = {s: _envelope(s, train, np.ones(len(train), bool)) for s in NEIGHBOURS}
        envs["DesertGardens"] = _envelope("DesertGardens", train, train["U"].to_numpy(), uncapped=True)
        cs = pd.DataFrame({s: envs[s].calculate_clear_sky_power(df.index, df[s]).to_numpy()
                           for s in envs}, index=df.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            kci = pd.DataFrame({s: (df[s] / cs[s]).where(cs[s] > 5).clip(0, 1.5) for s in NEIGHBOURS})
            kc_all = (df["neigh_sum"] / cs[NEIGHBOURS].sum(axis=1)).clip(0, 1.5)
            kc_dg = (df["DesertGardens"] / cs["DesertGardens"]).where(cs["DesertGardens"] > 5)
        kci_s = kci.rolling(3, center=True, min_periods=1).mean()

        tr = (df.index.year != yr) & df["U"].to_numpy()
        te = df.index.year == yr
        fitted = {}
        for name, K in [("kt_weighted", kci), ("kt_wsmooth", kci_s)]:
            ok = tr & K.notna().all(axis=1).to_numpy() & kc_dg.notna().to_numpy()
            w, _ = nnls(K[ok].to_numpy(), kc_dg[ok].to_numpy())
            w = w / w.sum() if w.sum() > 0 else np.full(len(NEIGHBOURS), 1 / len(NEIGHBOURS))
            fitted[name] = w
            kc_hat = (K.apply(lambda col: col.fillna(kc_all)) * w).sum(axis=1)
            pred[name][te] = (kc_hat * cs["DesertGardens"])[te].to_numpy()
        weights[int(yr)] = {k: dict(zip(NEIGHBOURS, np.round(v, 3))) for k, v in fitted.items()}
        pred["kt_sum"][te] = (kc_all * cs["DesertGardens"])[te].to_numpy()
        env_dg[te] = cs["DesertGardens"][te].to_numpy()
        kc_sum[te] = kc_all[te].to_numpy()
        print(f"  year {yr}: weights (smoothed) {weights[int(yr)]['kt_wsmooth']}", flush=True)

    kc_std = kc_sum.rolling(4, center=True, min_periods=2).std().fillna(0)
    df["sky"] = sky_class(kc_sum.to_numpy(), kc_std.to_numpy())
    U = df["U"].to_numpy()
    mid = (~U) & (df.index.hour >= 11) & (df.index.hour < 14)

    report = {"weights_by_heldout_year": weights}
    for v in variants:
        p = pred[v].to_numpy()
        y = df["DesertGardens"].to_numpy()
        m = metrics(y[U], p[U], y[U])
        by_sky = {c: metrics(y[U & (df["sky"] == c).to_numpy()], p[U & (df["sky"] == c).to_numpy()],
                             y[U & (df["sky"] == c).to_numpy()]).get("all") for c in SKY_CLASSES}
        exceed = float(np.mean(y[mid] > 1.05 * p[mid] + 5))
        report[v] = {"by_level": m, "by_sky": by_sky, "midday_violations": round(exceed, 4)}
        print(f"\n[{v}] midday violations {exceed:.2%}")
        for k, val in m.items():
            print(f"   {k:14s} {val}")
        for c, val in by_sky.items():
            print(f"   sky={c:9s} {val}")

    best = min(variants, key=lambda v: (report[v]["midday_violations"] >= 0.10,
                                        report[v]["by_level"]["DG_400-2000kW"]["MAE"]))
    print(f"\nselected: {best}")
    avail = pred[best].clip(lower=0)

    # conformal intervals: relative residual, Mondrian by sky class, calibrated out-of-year
    rel = (df["DesertGardens"] - avail) / env_dg.where(env_dg > 20)
    lo = pd.Series(np.nan, index=df.index)
    hi = pd.Series(np.nan, index=df.index)
    cover = {c: [] for c in SKY_CLASSES}
    width = {c: [] for c in SKY_CLASSES}
    for yr in years:
        cal = U & (df.index.year != yr)
        te = df.index.year == yr
        for c in SKY_CLASSES:
            sc = df["sky"].to_numpy() == c
            q_lo = conformal_q(-rel[cal & sc].to_numpy(), ALPHA / 2)   # lower tail
            q_hi = conformal_q(rel[cal & sc].to_numpy(), ALPHA / 2)    # upper tail
            rows = te & sc
            lo[rows] = (avail - q_lo * env_dg)[rows].clip(lower=0)
            hi[rows] = (avail + q_hi * env_dg)[rows]
            ev = rows & U
            if ev.sum():
                y = df["DesertGardens"][ev]
                cover[c].append(((y >= lo[ev]) & (y <= hi[ev])).to_numpy())
                width[c].append((hi[ev] - lo[ev]).to_numpy())
    cov_report = {c: {"coverage": round(float(np.concatenate(cover[c]).mean()), 3),
                      "mean_width_kW": round(float(np.concatenate(width[c]).mean()), 1),
                      "n": int(sum(len(a) for a in cover[c]))} for c in SKY_CLASSES if cover[c]}
    print(f"\nconformal {int(100 * (1 - ALPHA))}% intervals, leave-one-year-out coverage on U:")
    for c, v in cov_report.items():
        print(f"   {c:9s} {v}")

    delivered = df["DesertGardens"]
    share = lambda a: float((a - delivered).clip(lower=0).sum() / a.sum())  # noqa: E731
    shares = {"point": share(avail), "lower": share(lo.fillna(avail)), "upper": share(hi.fillna(avail))}
    print("curtailed share of DG available energy: point {point:.1%} (interval {lower:.1%} - {upper:.1%})".format(**shares))
    report.update({"selected": best, "conformal": cov_report,
                   "curtailed_energy_share": {k: round(v, 4) for k, v in shares.items()}})

    if args.out_csv:
        out = pd.DataFrame({"DG_delivered": delivered, "DG_available": avail, "DG_available_lo90": lo,
                            "DG_available_hi90": hi, "DG_clear_sky": env_dg, "neighbour_kc": kc_sum,
                            "sky": df["sky"], "U": df["U"], "Total_delivered": df["Total"], "GHI": df["GHI"]})
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index_label="timestamp")
        print("series ->", args.out_csv)
    if args.metrics:
        args.metrics.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
