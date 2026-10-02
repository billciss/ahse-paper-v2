"""
compare_nwp_runs.py
===================
Like-for-like comparison of two main.py runs on the same data and split (e.g. without and with
GFS covariates): metrics side by side, AHSE block decisions, and the significance of the MAE
difference of selected models.

Forecasts issued on the same day overlap strongly, so losses are first averaged per issue day
(daytime targets only); the test then uses
  - a bootstrap over days of the MAE difference (95% interval),
  - a Diebold-Mariano statistic on the daily loss differentials with a Newey-West variance.

Usage:
    python scripts/compare_nwp_runs.py --a results/yulara_neighbours_2018 --b results/yulara_neighbours_2018_gfs \
        --horizon 1h --out results/yulara_neighbours_2018_gfs/tables/comparison_no_gfs_vs_gfs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def daily_losses(y, p, mask, day):
    err = np.where(mask, np.abs(y - p), 0.0).sum(axis=1)
    cnt = mask.sum(axis=1)
    uniq, inv = np.unique(day, return_inverse=True)
    s_err, s_cnt = np.bincount(inv, err), np.bincount(inv, cnt)
    ok = s_cnt > 0
    return s_err[ok], s_cnt[ok]


def compare(y, mask, day, p_a, p_b, n_boot=5000, seed=0, lags=None):
    ea, cnt = daily_losses(y, p_a, mask, day)
    eb, _ = daily_losses(y, p_b, mask, day)
    mae_a, mae_b = ea.sum() / cnt.sum(), eb.sum() / cnt.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), (n_boot, len(ea)))
    diff = (ea[idx].sum(1) - eb[idx].sum(1)) / cnt[idx].sum(1)
    d = ea / cnt - eb / cnt                       # daily MAE differential (A - B)
    n = len(d)
    lags = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    dm = d.mean() / np.sqrt(var / n)
    return {"mae_a": float(mae_a), "mae_b": float(mae_b), "gain_b_over_a": float(mae_a - mae_b),
            "gain_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "dm_stat": float(dm), "dm_p_two_sided": float(2 * stats.norm.sf(abs(dm))),
            "n_days": int(n), "nw_lags": lags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--horizon", default="1h")
    ap.add_argument("--models", nargs="+", default=["AHSE", "AHSE_global", "LightGBM", "XGBoost"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    za = np.load(args.a / "tables" / f"predictions_{args.horizon}.npz")
    zb = np.load(args.b / "tables" / f"predictions_{args.horizon}.npz")
    for k in ("y_test", "daytime_mask_test", "target_start_rows_test"):
        if not np.array_equal(za[k], zb[k]):
            raise SystemExit(f"runs are not comparable: {k} differs")
    y, mask = za["y_test"], za["daytime_mask_test"].astype(bool)
    per_day = int(pd.Timedelta("1D") / pd.Timedelta(args.horizon))
    day = (za["target_start_rows_test"] - 1) // per_day

    ma = pd.read_csv(args.a / "tables" / f"metrics_global_{args.horizon}.csv").set_index("Model")
    mb = pd.read_csv(args.b / "tables" / f"metrics_global_{args.horizon}.csv").set_index("Model")
    cols = [c for c in ["MAE", "RMSE", "R2", "Skill_Score_Daily"] if c in ma.columns]
    table = ma[cols].add_suffix("_A").join(mb[cols].add_suffix("_B"), how="outer").sort_values("MAE_B")
    print(table.round(4).to_string())

    tests = {}
    for m in args.models:
        if f"test__{m}" in za.files and f"test__{m}" in zb.files:
            tests[f"{m}: A vs B"] = compare(y, mask, day, za[f"test__{m}"], zb[f"test__{m}"])
    if "test__AHSE" in zb.files:
        best_a = ma[cols].drop(index=[i for i in ma.index if i.startswith("AHSE")], errors="ignore")["MAE"].idxmin()
        tests[f"{best_a} (A) vs AHSE (B)"] = compare(y, mask, day, za[f"test__{best_a}"], zb["test__AHSE"])
    for name, t in tests.items():
        print(f"{name}: MAE {t['mae_a']:.4f} -> {t['mae_b']:.4f}, gain {t['gain_b_over_a']:+.4f} "
              f"[{t['gain_ci95'][0]:+.4f}, {t['gain_ci95'][1]:+.4f}], DM {t['dm_stat']:.2f} (p={t['dm_p_two_sided']:.3g})")

    dec = {}
    for tag, run in (("A", args.a), ("B", args.b)):
        f = run / "tables" / f"ahse_block_decisions_{args.horizon}.json"
        if f.exists():
            dec[tag] = json.loads(f.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out.with_suffix(".csv"))
    args.out.with_suffix(".json").write_text(json.dumps({"tests": tests, "block_decisions": dec}, indent=1))
    for tag, d in dec.items():
        print(tag, "reference", d["reference"], {k: v["chosen"] for k, v in d["decisions"].items()})


if __name__ == "__main__":
    main()
