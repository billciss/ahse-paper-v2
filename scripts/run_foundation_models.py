"""
run_foundation_models.py
========================
Zero-shot evaluation of time-series foundation models on EXACTLY the test samples used by
the AHSE pipeline (same splits, gap exclusion and daylight filter, via
target_start_rows returned by the preprocessor), in k_t space with the same daytime mask
and skill-score definitions as main.py.

Variants (Chronos-2, amazon/chronos-2):
  chronos2_ctx1d      : k_t history of one day (same information as the pipeline lookback)
  chronos2_ctx7d      : k_t history of seven days
  chronos2_ctx7d_cov  : seven days + known-future covariates (solar zenith angle and
                        clear-sky GHI, both deterministic, no leakage)
Reference baselines recomputed on the same samples: last-value and day-ahead persistence.

Usage (WSL):
    python scripts/run_foundation_models.py --data-config configs/data_config_yulara_neighbours.yaml \
        --data data/processed/yulara_neighbours_1h.csv --horizon 1h --out-dir results/yulara_neighbours
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_engine.preprocessor import MultiHorizonPreprocessor  # noqa: E402
from src.evaluation.metrics_factory import MetricsFactory  # noqa: E402

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
COVARIATES = ["zenith_angle", "clear_sky_ghi"]


def prepare(data_path, config_path, horizon):
    cfg = yaml.safe_load(open(config_path))
    h_cfg = next(h for h in cfg["forecasting"]["available_horizons"] if h["name"] == horizon)
    pre = MultiHorizonPreprocessor(config_path=config_path)
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    out = pre.prepare_multi_horizon_data(
        df, horizon=horizon, lookback_window=h_cfg["lookback"], forecast_horizon=h_cfg["horizon"],
        target_column=cfg["preprocessing"]["target_column"],
        train_ratio=cfg["data_split"]["train_ratio"], val_ratio=cfg["data_split"]["val_ratio"])
    # covariates on the same prepared frame (solar geometry recomputed by the physics step)
    frame = pre.clear_sky_calc.add_physics_features(
        pre._validate_and_prepare(df, cfg["preprocessing"]["target_column"]),
        power_column=cfg["preprocessing"]["target_column"], epsilon=pre.kt_epsilon,
        kt_max=pre.kt_clip_max, fit_fraction=cfg["data_split"]["train_ratio"])
    assert len(frame) == len(out["target_series"]), "covariate frame not aligned with the target series"
    cov = {c: frame[c].to_numpy(dtype=np.float32) for c in COVARIATES}
    cov["clear_sky_ghi"] = cov["clear_sky_ghi"] / 1000.0
    return out, cov, h_cfg["horizon"]


def build_inputs(series, cov, starts, ctx, H, with_cov):
    items = []
    for s in starts:
        a = max(0, s - ctx)
        item = {"target": series[a:s].astype(np.float32)}
        if with_cov:
            item["past_covariates"] = {c: cov[c][a:s] for c in COVARIATES}
            item["future_covariates"] = {c: cov[c][s:s + H] for c in COVARIATES}
        items.append(item)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--horizon", default="1h", choices=["15min", "1h"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from chronos import Chronos2Pipeline

    out, cov, H = prepare(args.data, args.data_config, args.horizon)
    series, starts, y = out["target_series"].astype(np.float32), out["target_start_rows"]["test"], out["y_test"]
    mask = out["daytime_mask"]
    per_day = int(pd.Timedelta("1D") / pd.Timedelta(args.horizon))
    print(f"test samples {len(starts):,}, H={H}, steps/day={per_day}")

    preds = {"Persistence": np.repeat(series[starts - 1][:, None], H, axis=1),
             "SmartPersistence": np.stack([series[s - per_day:s - per_day + H] for s in starts])}
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)
    quantiles_store, timing = {}, {}
    for name, ctx, with_cov in [("chronos2_ctx1d", per_day, False),
                                ("chronos2_ctx7d", 7 * per_day, False),
                                ("chronos2_ctx7d_cov", 7 * per_day, True)]:
        t0 = time.time()
        q_all, m_all = [], []
        inputs = build_inputs(series, cov, starts, ctx, H, with_cov)
        for i in range(0, len(inputs), args.batch_size):
            with torch.no_grad():
                q, m = pipe.predict_quantiles(inputs[i:i + args.batch_size], prediction_length=H,
                                              quantile_levels=QUANTILES)
            q_all += [t[0].cpu().numpy() for t in q]
            m_all += [t[0].cpu().numpy() for t in m]
        q_arr = np.clip(np.stack(q_all), 0, 1.5)
        preds[name] = q_arr[:, :, QUANTILES.index(0.5)]
        quantiles_store[name] = q_arr
        timing[name] = round(time.time() - t0, 1)
        print(f"  {name}: {timing[name]} s", flush=True)

    rows = []
    for name, p in preds.items():
        m = MetricsFactory.compute_all(y, p, mask, y_persistence=preds["Persistence"],
                                       y_smart_persistence=preds["SmartPersistence"])
        m["Model"] = name
        rows.append(m)
    table = pd.DataFrame(rows).sort_values("R2", ascending=False)
    print(table[["Model", "RMSE", "MAE", "R2", "Skill_Score", "Skill_Score_Daily"]].to_string(index=False))

    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / f"metrics_foundation_{args.horizon}.csv", index=False)
    np.savez_compressed(tables / f"predictions_foundation_{args.horizon}.npz", y_test=y, daytime_mask=mask,
                        target_start_rows=starts, quantile_levels=np.array(QUANTILES),
                        **{f"median__{k}": v for k, v in preds.items()},
                        **{f"quantiles__{k}": v for k, v in quantiles_store.items()})
    (tables / f"foundation_run_{args.horizon}.json").write_text(json.dumps(
        {"model": "amazon/chronos-2", "device": args.device, "seconds": timing,
         "n_test": int(len(starts)), "horizon_steps": int(H)}, indent=1))


if __name__ == "__main__":
    main()
