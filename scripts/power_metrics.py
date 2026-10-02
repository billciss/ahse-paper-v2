"""
power_metrics.py
================
Recomputes the test metrics of cached main.py runs in POWER units (kW and % of the DC capacity):
each model's k_t forecasts are multiplied by the clear-sky power envelope P_cs of the run's
configuration and compared with the measured power, on the run's daytime targets. P_cs depends
only on the data and the configuration (not on the seed), so it is computed once per site.

Per run: tables/metrics_power_<h>.csv. Per arm (several seeds): mean and standard deviation.
Paired tests (day-level bootstrap + Diebold-Mariano, scripts/compare_nwp_runs.compare):
AHSE without vs with GFS for every seed, and AHSE vs the best single model of each run.

Usage:
    python scripts/power_metrics.py --out results/power_summary
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_nwp_runs import compare  # noqa: E402
from src.data_engine.preprocessor import load_and_prepare_multi_horizon  # noqa: E402

SITES = {
    "yulara": dict(data="data/processed/yulara_neighbours_1h.csv",
                   config="configs/data_config_yulara_neighbours.yaml", capacity=768.0,
                   arms={"no_gfs": ["results/yulara_neighbours"] + [f"results/seeds_gfs/seed_{s}/no_gfs" for s in (1, 2, 3, 4)],
                         "gfs": ["results/yulara_neighbours_gfs"] + [f"results/seeds_gfs/seed_{s}/gfs" for s in (1, 2, 3, 4)]}),
    "nist": dict(data="data/processed/nist_pvdaq_1h.csv", config="configs/data_config_nist.yaml",
                 capacity=271.0, arms={"no_gfs": ["results/nist"], "gfs": ["results/nist_gfs"]}),
}
HORIZON = "1h"
SINGLE_EXCLUDE = ("AHSE", "AHSE_global", "Persistence", "SmartPersistence", "GFS")


def site_arrays(site):
    d = load_and_prepare_multi_horizon(site["data"], config_path=site["config"], horizon=HORIZON)
    target = yaml.safe_load(open(site["config"]))["preprocessing"]["target_column"]
    power = pd.read_csv(site["data"], index_col=0, parse_dates=True)[target].to_numpy()
    return np.asarray(d["clear_sky_power"], dtype=float), power, d["target_start_rows"]["test"]


def run_metrics(run, pcs, power, starts, capacity):
    z = np.load(Path(run) / "tables" / f"predictions_{HORIZON}.npz")
    if not np.array_equal(z["target_start_rows_test"], starts):
        raise SystemExit(f"{run}: cached test instants differ from the configuration's")
    rows = starts[:, None] + np.arange(z["y_test"].shape[1])[None, :]
    p_cs, p_meas = pcs[rows], power[rows]
    mask = z["daytime_mask_test"].astype(bool) & np.isfinite(p_meas)
    preds = {k.split("__", 1)[1]: z[k] * p_cs for k in z.files if k.startswith("test__")}
    y = p_meas[mask]
    ref = preds["SmartPersistence"][mask]
    rows_out = []
    for m, p in preds.items():
        e = p[mask] - y
        rows_out.append({"Model": m, "MAE_kW": np.abs(e).mean(), "RMSE_kW": np.sqrt((e ** 2).mean()),
                         "nMAE_pct": 100 * np.abs(e).mean() / capacity,
                         "nRMSE_pct": 100 * np.sqrt((e ** 2).mean()) / capacity,
                         "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                         "Skill_vs_dayahead": 1 - (e ** 2).mean() / ((ref - y) ** 2).mean()})
    day = (starts - 1) // int(pd.Timedelta("1D") / pd.Timedelta(HORIZON))
    return pd.DataFrame(rows_out).sort_values("MAE_kW"), preds, p_meas, mask, day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tests, summaries = {}, []
    for name, site in SITES.items():
        pcs, power, starts = site_arrays(site)
        cache = {}
        for arm, runs in site["arms"].items():
            frames = []
            for run in runs:
                table, preds, p_meas, mask, day = run_metrics(run, pcs, power, starts, site["capacity"])
                table.to_csv(Path(run) / "tables" / f"metrics_power_{HORIZON}.csv", index=False)
                frames.append(table.assign(run=run))
                cache[run] = preds
                singles = table[~table.Model.isin(SINGLE_EXCLUDE)]
                best = singles.iloc[0].Model
                t = compare(p_meas, mask, day, preds[best], preds["AHSE"])
                tests[f"{name} {arm} {run}: {best} (best single) vs AHSE"] = t
            df = pd.concat(frames)
            agg = df.groupby("Model")[["MAE_kW", "nMAE_pct", "RMSE_kW", "nRMSE_pct", "R2", "Skill_vs_dayahead"]] \
                .agg(["mean", "std"]).round(4)
            agg.columns = [f"{a}_{b}" for a, b in agg.columns]
            agg = agg.sort_values("MAE_kW_mean")
            agg.insert(0, "n_runs", df.groupby("Model").size())
            agg.to_csv(args.out / f"{name}_{arm}_power_{HORIZON}.csv")
            summaries.append((name, arm, agg))
        for a, b in zip(site["arms"]["no_gfs"], site["arms"]["gfs"]):
            _, _, p_meas, mask, day = run_metrics(a, pcs, power, starts, site["capacity"])
            t = compare(p_meas, mask, day, cache[a]["AHSE"], cache[b]["AHSE"])
            tests[f"{name} AHSE no_gfs vs gfs: {a} -> {b}"] = t
    (args.out / f"paired_tests_power_{HORIZON}.json").write_text(json.dumps(tests, indent=1))
    pd.set_option("display.width", 200)
    for name, arm, agg in summaries:
        print(f"\n== {name} / {arm}")
        print(agg[["n_runs", "MAE_kW_mean", "MAE_kW_std", "nMAE_pct_mean", "RMSE_kW_mean", "R2_mean",
                   "Skill_vs_dayahead_mean"]].to_string())
    print()
    for k, t in tests.items():
        print(f"{k}: MAE {t['mae_a']:.2f} -> {t['mae_b']:.2f} kW, gain {t['gain_b_over_a']:+.2f} "
              f"[{t['gain_ci95'][0]:+.2f}, {t['gain_ci95'][1]:+.2f}], DM {t['dm_stat']:.2f}, p={t['dm_p_two_sided']:.2g}")


if __name__ == "__main__":
    main()
