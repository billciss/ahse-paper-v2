"""
compare_runs_power.py
=====================
Compares two main.py runs on the same test forecasts in POWER units, for runs whose clearness
index differs (e.g. chronological vs interleaved validation: the clear-sky power envelope P_cs is
fitted on different training rows, so the k_t targets differ slightly). Each run's k_t forecasts
are turned into power with its own P_cs and compared with the measured power, on the targets
that are daytime for both runs; significance as in compare_nwp_runs.py (day-level bootstrap and
Diebold-Mariano with Newey-West variance).

Usage:
    python scripts/compare_runs_power.py --data data/processed/nist_pvdaq_1h.csv \
        --a results/nist --a-config configs/data_config_nist.yaml \
        --b results/nist_seasonal --b-config configs/data_config_nist_interleaved.yaml \
        --capacity 271 --out results/nist_seasonal/tables/comparison_chrono_vs_seasonal_power
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--a-config", required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--b-config", required=True)
    ap.add_argument("--horizon", default="1h")
    ap.add_argument("--capacity", type=float, required=True, help="kW, to normalise the MAE")
    ap.add_argument("--models", nargs="+", default=["AHSE", "AHSE_global", "XGBoost", "LightGBM"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    za = np.load(args.a / "tables" / f"predictions_{args.horizon}.npz")
    zb = np.load(args.b / "tables" / f"predictions_{args.horizon}.npz")
    if not np.array_equal(za["target_start_rows_test"], zb["target_start_rows_test"]):
        raise SystemExit("the two runs do not forecast the same test instants")
    starts = za["target_start_rows_test"]
    rows = starts[:, None] + np.arange(za["y_test"].shape[1])[None, :]

    pcs = {}
    for tag, cfg in (("a", args.a_config), ("b", args.b_config)):
        d = load_and_prepare_multi_horizon(args.data, config_path=cfg, horizon=args.horizon)
        pcs[tag] = np.asarray(d["clear_sky_power"], dtype=float)[rows]
        if not np.array_equal(d["target_start_rows"]["test"], starts):
            raise SystemExit(f"config {cfg} does not reproduce the cached test instants")
    target = yaml.safe_load(open(args.a_config))["preprocessing"]["target_column"]
    p_meas = pd.read_csv(args.data, index_col=0, parse_dates=True)[target].to_numpy()[rows]
    mask = za["daytime_mask_test"].astype(bool) & zb["daytime_mask_test"].astype(bool) & np.isfinite(p_meas)
    day = (starts - 1) // int(pd.Timedelta("1D") / pd.Timedelta(args.horizon))

    results = {}
    print(f"{'model':12s} {'MAE A':>8s} {'MAE B':>8s}  (kW; % of {args.capacity:.0f} kW)   gain B over A [95% CI], DM, p")
    for m in args.models:
        if f"test__{m}" not in za.files or f"test__{m}" not in zb.files:
            continue
        pa, pb = za[f"test__{m}"] * pcs["a"], zb[f"test__{m}"] * pcs["b"]
        t = compare(p_meas, mask, day, pa, pb)
        t["mae_a_pct"], t["mae_b_pct"] = 100 * t["mae_a"] / args.capacity, 100 * t["mae_b"] / args.capacity
        results[m] = t
        print(f"{m:12s} {t['mae_a']:8.2f} {t['mae_b']:8.2f}  ({t['mae_a_pct']:.2f}% -> {t['mae_b_pct']:.2f}%)   "
              f"{t['gain_b_over_a']:+.2f} [{t['gain_ci95'][0]:+.2f}, {t['gain_ci95'][1]:+.2f}], "
              f"{t['dm_stat']:.2f}, p={t['dm_p_two_sided']:.3g}")
    # within each run: AHSE against the best single tree model, in power
    for tag, z in (("A", za), ("B", zb)):
        p = pcs[tag.lower()]
        for ref in ("XGBoost", "LightGBM"):
            if f"test__{ref}" in z.files and "test__AHSE" in z.files:
                t = compare(p_meas, mask, day, z[f"test__{ref}"] * p, z["test__AHSE"] * p)
                results[f"{tag}: {ref} vs AHSE"] = t
                print(f"run {tag}: {ref} -> AHSE  MAE {t['mae_a']:.2f} -> {t['mae_b']:.2f} kW, "
                      f"gain {t['gain_b_over_a']:+.2f} [{t['gain_ci95'][0]:+.2f}, {t['gain_ci95'][1]:+.2f}], "
                      f"DM {t['dm_stat']:.2f}, p={t['dm_p_two_sided']:.3g}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
