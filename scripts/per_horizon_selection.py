"""
per_horizon_selection.py
========================
Post-hoc report of the honest per-horizon-block selection (src/models/honest_selection.py,
the same code used by main.py for AHSE) from the plain prediction cache written by main.py
(results/<run>/tables/predictions_<h>.npz).

It compares, on the test set: the best single model (validation reference), the original
single-strategy selector (AHSE_global, if present), the block selector, and the per-block
oracle (best candidate on TEST, reported only as an upper bound, not as a method).

Usage (WSL):
    python scripts/per_horizon_selection.py --cache results/yulara_neighbours/tables/predictions_1h.npz \
        --steps-per-hour 1 --out results/yulara_neighbours/tables/per_horizon_selection_1h.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.metrics_factory import MetricsFactory  # noqa: E402
from src.models.honest_selection import HonestBlockSelector, default_blocks  # noqa: E402


def masked_mae(y, p, m):
    return float(np.abs(y - p)[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--steps-per-hour", type=int, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    z = np.load(args.cache)

    yv, yt = z["y_val"], z["y_test"]
    mv = z["daytime_mask_val"] if "daytime_mask_val" in z.files else yv > 0
    mt = z["daytime_mask_test"] if "daytime_mask_test" in z.files else yt > 0
    pv = {k[5:]: z[k] for k in z.files if k.startswith("val__")}
    pt = {k[6:]: z[k] for k in z.files if k.startswith("test__")}
    blocks = default_blocks(yv.shape[1], args.steps_per_hour)
    # day blocks for the bootstrap: approximate days from sample order (main.py uses dates)
    days = np.arange(len(yv)) // (24 * args.steps_per_hour)

    sel = HonestBlockSelector(blocks).fit(pv, yv, mv, days)
    sel_test = sel.predict(pt)

    # per-block oracle on TEST (upper bound only)
    models = sorted(m for m in pv if m not in sel.exclude)
    oracle = np.array(pt[sel.reference], dtype=float)
    for name, (a, b) in blocks.items():
        sl = slice(a, b)
        rank = sorted(models, key=lambda m: masked_mae(yv[:, sl], pv[m][:, sl], mv[:, sl]))
        cands = [(m,) for m in models] + [tuple(rank[:k]) for k in sel.top_k if k <= len(rank)]
        best = min(cands, key=lambda c: masked_mae(yt[:, sl], np.mean([pt[m] for m in c], 0)[:, sl], mt[:, sl]))
        oracle[:, sl] = np.mean([pt[m] for m in best], 0)[:, sl]
        sel.decisions[name]["oracle_on_test"] = list(best)

    smart = pt.get("SmartPersistence")
    rows = {}
    for name, p in [(f"reference_best_single({sel.reference})", pt[sel.reference]),
                    ("AHSE_global", pt.get("AHSE_global")), ("per_horizon_selection", sel_test),
                    ("per_block_oracle (upper bound)", oracle)]:
        if p is None:
            continue
        m = MetricsFactory.compute_all(yt, p, mt, y_smart_persistence=smart)
        rows[name] = {"MAE": round(m["MAE"], 5), "RMSE": round(m["RMSE"], 5), "R2": round(m["R2"], 4),
                      "Skill_Score_Daily": round(m["Skill_Score_Daily"], 4),
                      **{f"MAE_{b}": round(masked_mae(yt[:, a:c], p[:, a:c], mt[:, a:c]), 5)
                         for b, (a, c) in blocks.items()}}
    report = {"reference": sel.reference, "decisions": sel.decisions, "test": rows}
    print(json.dumps(report, indent=1, default=float))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
