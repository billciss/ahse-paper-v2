"""
censored_available_power.py
===========================
Available-power reconstruction for Yulara Desert Gardens (DG) treating curtailment as
CENSORING: delivered = min(available, cap), so delivered <= available and instants where
the cap binds carry no information on the available level.

Lessons from refine_available_power.py: "total < 350 kW" and "after 16 h" do not
guarantee absence of curtailment (clear-sky U instants at 9-15 h were curtailed by
80-280 kW; curtailment persists at 16 h in the cool season), and DG responds more
strongly than its neighbours to diffuse light (overcast bias).

Method, fitted per held-out year on the other years (leave-one-year-out):
  U0  = sun up and (hour < 9 or hour >= 17 or (total < 350 kW and neighbour k_c < 0.5))
  repeat 3 times:
     DG clear-sky envelope fitted on U_k (plane-of-array, uncapped)
     neighbour weights w (NNLS, sum 1) and a monotone map g (isotonic) on U_k:
         DG / DG_envelope  ~  g( sum_i w_i k_c,i )
     U_{k+1} = U0 and delivered >= 0.85 * available   (drop likely-curtailed points)
Outages (see yulara_data_quality.py): a neighbour is masked when its index is < 0.02 or
< 30% of the row median under GHI > 300 W/m2, and the neighbour index is computed from the
available neighbours only (weights renormalised). DG instants with near-zero output under
sun while neighbours produce are flagged as DG outages, excluded from training and from
the curtailment statistics (observed curtailment never goes to zero).
Evaluation on a FIXED set independent of the trimming: held-out year, 07:00-08:59.
Uncertainty: split-conformal 90% intervals on the relative residual, Mondrian by sky
class, calibrated on the fixed set of the OTHER years (OOF residuals).
Limitation: no uncurtailed clear-sky midday instants exist, so midday accuracy is only
checked indirectly (delivered should not exceed available).

Usage (WSL):
    python scripts/censored_available_power.py --downloads /mnt/c/Users/billc/Downloads \
        --out-csv data/processed/yulara_dg_available_15min.csv \
        --metrics results/curtailment/censored_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_available_power import NEIGHBOURS, _envelope, build_frame, metrics  # noqa: E402
from refine_available_power import SKY_CLASSES, conformal_q, sky_class  # noqa: E402

ALPHA = 0.10
TRIM = 0.85
N_ITER = 3
DG_NAMEPLATE_KW = 1058.4


def mask_outages(kci, ghi):
    """NaN-out neighbour indices during (partial) outages under clear enough skies."""
    sunny = (ghi > 300).to_numpy()[:, None]
    med = kci.median(axis=1).to_numpy()[:, None]
    bad = sunny & ((kci.to_numpy() < 0.02) | (kci.to_numpy() < 0.3 * med))
    return kci.mask(bad)


def combine(kci, w, fallback):
    """Weighted neighbour index over the available neighbours, renormalising the weights."""
    avail_w = kci.notna().to_numpy() * w
    den = avail_w.sum(axis=1)
    num = np.nansum(kci.to_numpy() * w, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den >= 0.2, num / den, fallback.to_numpy())
    return pd.Series(out, index=kci.index)


def fold_model(df, train_mask, u0, cs_neigh, kci, kc_all, dg_ok):
    """Fit DG envelope, weights and isotonic map on the training rows (censoring-aware)."""
    U = u0 & train_mask & dg_ok
    for _ in range(N_ITER):
        rows = df[train_mask]
        env = _envelope("DesertGardens", rows, U[train_mask], uncapped=True)
        cs_dg = env.calculate_clear_sky_power(df.index, df["DesertGardens"])
        with np.errstate(divide="ignore", invalid="ignore"):
            kc_dg = (df["DesertGardens"] / cs_dg).where(cs_dg > 5)
        ok = U & kci.notna().all(axis=1).to_numpy() & kc_dg.notna().to_numpy()
        w, _ = nnls(kci[ok].to_numpy(), kc_dg[ok].to_numpy())
        w = w / w.sum() if w.sum() > 0 else np.full(len(NEIGHBOURS), 1 / len(NEIGHBOURS))
        kc_hat = combine(kci, w, kc_all)
        iso = IsotonicRegression(y_min=0, y_max=1.5, increasing=True, out_of_bounds="clip")
        iso.fit(kc_hat[ok].to_numpy(), kc_dg[ok].to_numpy())
        g = pd.Series(iso.predict(kc_hat.fillna(0).to_numpy()), index=df.index).where(kc_hat.notna())
        avail = g * cs_dg
        U = u0 & train_mask & dg_ok & (df["DesertGardens"] >= TRIM * avail).to_numpy()
    return avail, cs_dg, w, int(U.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path)
    ap.add_argument("--metrics", type=Path)
    args = ap.parse_args()

    df = build_frame(args.downloads)
    years = sorted(df.index.year.unique())
    hour = df.index.hour
    avail = pd.Series(np.nan, index=df.index)
    env_dg = pd.Series(np.nan, index=df.index)
    kc_sum = pd.Series(np.nan, index=df.index)
    info = {}

    for yr in years:
        train = (df.index.year != yr)
        te = ~train
        tr_rows = df[train]
        envs = {s: _envelope(s, tr_rows, np.ones(len(tr_rows), bool)) for s in NEIGHBOURS}
        cs_n = pd.DataFrame({s: envs[s].calculate_clear_sky_power(df.index, df[s]).to_numpy()
                             for s in NEIGHBOURS}, index=df.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            kci = pd.DataFrame({s: (df[s] / cs_n[s]).where(cs_n[s] > 5).clip(0, 1.5) for s in NEIGHBOURS})
            kci = mask_outages(kci, df["GHI"])
            on = kci.notna()
            kc_all = (df[NEIGHBOURS].where(on).sum(axis=1, min_count=1)
                      / cs_n.where(on).sum(axis=1, min_count=1)).clip(0, 1.5)
        dg_ok = ~((df["DesertGardens"] < 0.01 * DG_NAMEPLATE_KW) & (df["GHI"] > 300) & (kc_all > 0.3)).to_numpy()
        n_neigh_out = int((~on).any(axis=1)[te].sum())
        u0 = ((hour < 9) | (hour >= 17) | ((df["Total"] < 350) & (kc_all < 0.5))).to_numpy()
        a, cs_dg, w, n_u = fold_model(df, train, u0, cs_n, kci, kc_all, dg_ok)
        a[~dg_ok] = np.nan
        avail[te], env_dg[te], kc_sum[te] = a[te].to_numpy(), cs_dg[te].to_numpy(), kc_all[te].to_numpy()
        info[int(yr)] = {"weights": dict(zip(NEIGHBOURS, np.round(w, 3))), "n_train_uncurtailed": n_u,
                         "heldout_rows_with_neighbour_outage": n_neigh_out,
                         "heldout_dg_outage_rows": int((~dg_ok & te).sum())}
        print(f"  year {yr}: n_U {n_u:,}  weights {info[int(yr)]['weights']}  "
              f"neighbour-outage rows {n_neigh_out:,}  DG-outage rows {int((~dg_ok & te).sum()):,}", flush=True)

    avail = avail.clip(lower=0)
    kc_std = kc_sum.rolling(4, center=True, min_periods=2).std().fillna(0)
    df["sky"] = sky_class(kc_sum.to_numpy(), kc_std.to_numpy())
    y = df["DesertGardens"].to_numpy()
    valid = np.isfinite(avail.to_numpy())                       # excludes DG outages
    E = np.isin(hour, [7, 8]) & valid                           # fixed evaluation set
    mid = (hour >= 11) & (hour < 14) & valid
    ev = {"fixed_set_07_08h": metrics(y[E], avail.to_numpy()[E], y[E]),
          "by_sky_07_08h": {c: metrics(y[E & (df["sky"] == c).to_numpy()],
                                       avail.to_numpy()[E & (df["sky"] == c).to_numpy()],
                                       y[E & (df["sky"] == c).to_numpy()]).get("all") for c in SKY_CLASSES},
          "midday_violations": round(float(np.mean(y[mid] > 1.05 * avail.to_numpy()[mid] + 5)), 4)}
    print("\nfixed evaluation set (07-08 h, leave-one-year-out):")
    for k, v in ev["fixed_set_07_08h"].items():
        print(f"   {k:14s} {v}")
    for c, v in ev["by_sky_07_08h"].items():
        print(f"   sky={c:9s} {v}")
    print(f"   midday violations (delivered > 1.05*available + 5 kW): {ev['midday_violations']:.2%}")

    rel = (df["DesertGardens"] - avail) / env_dg.where(env_dg > 20)
    lo = pd.Series(np.nan, index=df.index)
    hi = pd.Series(np.nan, index=df.index)
    cover, width = {c: [] for c in SKY_CLASSES}, {c: [] for c in SKY_CLASSES}
    for yr in years:
        cal = E & (df.index.year != yr)
        te = (df.index.year == yr)
        for c in SKY_CLASSES:
            sc = (df["sky"] == c).to_numpy()
            q_lo = conformal_q(-rel[cal & sc].to_numpy(), ALPHA / 2)
            q_hi = conformal_q(rel[cal & sc].to_numpy(), ALPHA / 2)
            rows = te & sc
            lo[rows] = (avail - q_lo * env_dg)[rows].clip(lower=0)
            hi[rows] = (avail + q_hi * env_dg)[rows]
            chk = rows & E & np.isfinite(lo.to_numpy()) & np.isfinite(hi.to_numpy())
            if chk.sum():
                yy = df["DesertGardens"][chk]
                cover[c].append(((yy >= lo[chk]) & (yy <= hi[chk])).to_numpy())
                width[c].append((hi[chk] - lo[chk]).to_numpy())
    cov = {c: {"coverage": round(float(np.concatenate(cover[c]).mean()), 3),
               "mean_width_kW": round(float(np.concatenate(width[c]).mean()), 1),
               "n": int(sum(len(a) for a in cover[c]))} for c in SKY_CLASSES if cover[c]}
    print("\nconformal 90% intervals, coverage on the fixed set (leave-one-year-out):")
    for c, v in cov.items():
        print(f"   {c:9s} {v}")

    delivered = df["DesertGardens"].where(avail.notna())
    share = lambda a: float((a - delivered).clip(lower=0).sum() / a.sum())  # noqa: E731
    shares = {"point": share(avail), "lower": share(lo.fillna(avail)), "upper": share(hi.fillna(avail))}
    by_month = ((avail - delivered).clip(lower=0).groupby(df.index.month).sum()
                / avail.groupby(df.index.month).sum()).round(3).to_dict()
    print("curtailed share of DG available energy: point {point:.1%} (conservative bounds {lower:.1%} - {upper:.1%})"
          .format(**shares))
    print("   by month:", by_month)

    report = {"folds": info, "evaluation": ev, "conformal_fixed_set": cov,
              "curtailed_energy_share": {k: round(v, 4) for k, v in shares.items()},
              "curtailed_energy_share_by_month": {int(k): v for k, v in by_month.items()}}
    if args.metrics:
        args.metrics.write_text(json.dumps(report, indent=1, default=float))
    if args.out_csv:
        out = pd.DataFrame({"DG_delivered": delivered, "DG_available": avail, "DG_available_lo90": lo,
                            "DG_available_hi90": hi, "DG_clear_sky": env_dg, "neighbour_kc": kc_sum,
                            "sky": df["sky"], "Total_delivered": df["Total"], "GHI": df["GHI"]})
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index_label="timestamp")
        print("series ->", args.out_csv)


if __name__ == "__main__":
    main()
