"""
conformal_intervals.py
======================
Prediction intervals for the AHSE (block) forecast from a prediction cache written by main.py
(tables/predictions_<h>.npz).

The block selector is fitted on the validation set, so its validation residuals are optimistic.
Calibration therefore uses OUT-OF-FOLD validation forecasts: validation days are split into
contiguous folds, the selector is refitted without each fold and predicts it (cross-conformal,
in the spirit of CV+). The test forecast is the selector fitted on the whole validation set,
as in main.py. The test set is used only for evaluation.

Usage (WSL):
    python scripts/conformal_intervals.py --cache results/yulara_neighbours/tables/predictions_1h.npz \
        --horizon 1h --out results/yulara_neighbours/tables
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.conformal import MondrianConformal, adaptive_alphas, interval_metrics  # noqa: E402
from src.models.honest_selection import HonestBlockSelector, default_blocks  # noqa: E402


def load_cache(path):
    z = np.load(path)
    preds = {s: {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"{s}__")} for s in ("val", "test")}
    return z, preds


def regime_of_day(y, mask, day_ids):
    """Observed daily regime of each forecast row (evaluation only): mean daytime k_t of its targets."""
    m = np.where(mask, y, np.nan)
    mean_kt = np.nanmean(m, axis=1)
    return np.select([mean_kt >= 0.8, mean_kt >= 0.4], ["clear", "variable"], "overcast")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--horizon", default="1h")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.1])
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--key-model", default=None,
                    help="also bin by the forecast of this cached member (e.g. GFS), tertiles on validation")
    args = ap.parse_args()

    z, preds = load_cache(args.cache)
    y_val, y_test = z["y_val"], z["y_test"]
    m_val, m_test = z["daytime_mask_val"].astype(bool), z["daytime_mask_test"].astype(bool)
    step = pd.Timedelta(args.horizon)
    per_day = int(pd.Timedelta("1D") / step)
    # processed files hold a complete grid starting at midnight: row // per_day is the day;
    # a forecast belongs to the day it is issued (row of its last observation)
    day_val = (z["target_start_rows_val"] - 1) // per_day
    day_test = (z["target_start_rows_test"] - 1) // per_day
    blocks = default_blocks(y_val.shape[1], int(pd.Timedelta("1h") / step))
    base_val = {k: v for k, v in preds["val"].items() if not k.startswith("AHSE")}
    base_test = {k: v for k, v in preds["test"].items() if not k.startswith("AHSE")}

    # out-of-fold AHSE forecasts on validation (contiguous day folds)
    days = np.unique(day_val)
    oof = np.full(y_val.shape, np.nan)
    for fold in np.array_split(days, args.folds):
        hold = np.isin(day_val, fold)
        sel = HonestBlockSelector(blocks).fit({k: v[~hold] for k, v in base_val.items()},
                                              y_val[~hold], m_val[~hold], day_val[~hold])
        oof[hold] = sel.predict({k: v[hold] for k, v in base_val.items()})
    full = HonestBlockSelector(blocks).fit(base_val, y_val, m_val, day_val)
    p_test = full.predict(base_test)
    if "AHSE" in preds["test"]:
        gap = np.nanmax(np.abs(p_test - preds["test"]["AHSE"]))
        print(f"max |refitted - cached AHSE| on test: {gap:.2e}")

    cp = MondrianConformal(blocks).fit(y_val, oof, m_val)
    # alternative binning variable known at issue time: spread of the base models (std across
    # models), cut at the validation tertiles of daytime targets
    names = sorted(k for k in base_val if k != "Persistence")
    spread_val = np.std([base_val[k] for k in names], axis=0)
    spread_test = np.std([base_test[k] for k in names], axis=0)
    spread_edges = tuple(np.quantile(spread_val[m_val], [1 / 3, 2 / 3]))
    cp_spread = MondrianConformal(blocks, level_edges=spread_edges).fit(y_val, oof, m_val, key=spread_val)
    cp_key = None
    if args.key_model:
        key_val, key_test = preds["val"][args.key_model], preds["test"][args.key_model]
        key_edges = tuple(np.quantile(key_val[m_val], [1 / 3, 2 / 3]))
        cp_key = MondrianConformal(blocks, level_edges=key_edges).fit(y_val, oof, m_val, key=key_val)
    regime = regime_of_day(y_test, m_test, day_test)
    rows = []
    for alpha in args.alphas:
        lo, hi = cp.interval(p_test, alpha)
        variants = {"split_mondrian": (lo, hi)}

        def cover_fn(r, a):
            l, h = cp.interval(p_test[r], a)
            return (y_test[r] >= l) & (y_test[r] <= h)
        a_rows, _ = adaptive_alphas(cover_fn, m_test, day_test, alpha, gamma=args.gamma)
        lo_a, hi_a = cp.interval(p_test, a_rows)
        variants["aci_mondrian"] = (lo_a, hi_a)
        # pooled (non-Mondrian) split conformal for reference
        pooled = MondrianConformal({"all": (0, y_val.shape[1])}, level_edges=()).fit(y_val, oof, m_val)
        variants["split_pooled"] = pooled.interval(p_test, alpha)
        variants["split_spread"] = cp_spread.interval(p_test, alpha, key=spread_test)

        def cover_spread(r, a):
            l, h = cp_spread.interval(p_test[r], a, key=spread_test[r])
            return (y_test[r] >= l) & (y_test[r] <= h)
        a_rows_s, _ = adaptive_alphas(cover_spread, m_test, day_test, alpha, gamma=args.gamma)
        variants["aci_spread"] = cp_spread.interval(p_test, a_rows_s, key=spread_test)
        if cp_key is not None:
            variants[f"split_{args.key_model}"] = cp_key.interval(p_test, alpha, key=key_test)

        for name, (l, h) in variants.items():
            groups = {"all": m_test}
            for b, (a0, b0) in blocks.items():
                g = np.zeros_like(m_test)
                g[:, a0:b0] = True
                groups[f"block:{b}"] = m_test & g
            for r in ("clear", "variable", "overcast"):
                groups[f"regime:{r}"] = m_test & (regime == r)[:, None]
            for gname, gm in groups.items():
                if gm.any():
                    rows.append({"alpha": alpha, "nominal": 1 - alpha, "method": name, "group": gname,
                                 **interval_metrics(y_test, l, h, gm, alpha)})
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / f"conformal_{args.horizon}{args.tag}.csv"
    table.to_csv(out_csv, index=False)
    summary = table[table.group == "all"][["nominal", "method", "coverage", "width", "interval_score"]]
    print(summary.to_string(index=False))
    shown = ["split_mondrian", "split_spread"] + ([f"split_{args.key_model}"] if args.key_model else [])
    print(table[table.method.isin(shown) & (table.group != "all")]
          .pivot_table(index=["nominal", "group"], columns="method", values=["coverage", "width"])
          .round(3).to_string())
    (args.out / f"conformal_{args.horizon}{args.tag}.json").write_text(json.dumps({
        "cache": str(args.cache), "folds": args.folds, "gamma": args.gamma,
        "level_edges": list(cp.edges), "spread_edges": list(spread_edges), "spread_models": names,
        "oof_val_mae": float(np.abs(y_val - oof)[m_val].mean()),
        "selector_decisions": full.decisions}, indent=1, default=float))


if __name__ == "__main__":
    main()
