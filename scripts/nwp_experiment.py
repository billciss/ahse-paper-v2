"""
nwp_experiment.py
=================
Does adding archived GFS day-ahead forecasts improve the day-ahead PV forecast?

Same data, splits, gap exclusion and daylight filter as the AHSE pipeline (preprocessor),
restricted to the period where both GFS variables have been retrieved. Variants:
  lgbm_no_nwp : LightGBM on the pipeline features (flattened lookback window)
  lgbm_nwp    : same + for each of the H target hours the GFS 3-hour means of downward
                shortwave radiation and total cloud cover from the latest run PUBLISHED at the
                issue time (src/data_engine/nwp_features.py), the clear-sky GHI and the GFS
                clearness index below
  gfs_only    : GFS 3-hour radiation divided by the clear-sky GHI of the same 3-hour window,
                used directly as k_t forecast
  day-ahead persistence (reference for the skill score)
LightGBM settings are those of configs/model_params.yaml (TreeModelTrainer).

Usage (WSL):
    python scripts/nwp_experiment.py --data data/processed/yulara_neighbours_1h.csv \
        --data-config configs/data_config_yulara_neighbours.yaml --end 2018-12-31 \
        --gfs-dswrf data_external/gfs/gfs_yulara_DSWRF.csv --gfs-tcdc data_external/gfs/gfs_yulara_TCDC.csv \
        --out-dir results/yulara_neighbours_nwp
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_engine.nwp_features import gfs_target_features, load_gfs  # noqa: E402
from src.data_engine.preprocessor import MultiHorizonPreprocessor  # noqa: E402
from src.evaluation.metrics_factory import MetricsFactory  # noqa: E402
from src.training.trainer import TreeModelTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--model-config", default=str(ROOT / "configs" / "model_params.yaml"))
    ap.add_argument("--horizon", default="1h")
    ap.add_argument("--end", required=True, help="last date covered by both GFS variables")
    ap.add_argument("--gfs-dswrf", required=True)
    ap.add_argument("--gfs-tcdc", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data_config))
    h_cfg = next(h for h in cfg["forecasting"]["available_horizons"] if h["name"] == args.horizon)
    L, H = h_cfg["lookback"], h_cfg["horizon"]
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    df = df[df.index < pd.Timestamp(args.end).tz_localize(df.index.tz) + pd.Timedelta(days=1)] \
        if df.index.tz is not None else df[df.index < pd.Timestamp(args.end) + pd.Timedelta(days=1)]
    pre = MultiHorizonPreprocessor(config_path=args.data_config)
    d = pre.prepare_multi_horizon_data(df, horizon=args.horizon, lookback_window=L, forecast_horizon=H,
                                       target_column=cfg["preprocessing"]["target_column"],
                                       train_ratio=cfg["data_split"]["train_ratio"],
                                       val_ratio=cfg["data_split"]["val_ratio"])
    tidx = pd.DatetimeIndex(d["time_index"])
    step = pd.Timedelta(args.horizon)
    loc = pvlib.location.Location(cfg["location"]["latitude"], cfg["location"]["longitude"],
                                  tz=cfg["location"]["timezone"], altitude=cfg["location"]["altitude"])
    gfs = load_gfs(args.gfs_dswrf, args.gfs_tcdc)
    print(f"GFS runs {gfs['init'].nunique():,} from {gfs['init'].min()} to {gfs['init'].max()}")

    feats, cov_stats = {}, {}
    for split in ["train", "val", "test"]:
        f = gfs_target_features(gfs, tidx, d["target_start_rows"][split], H, step, loc)
        feats[split] = {"dswrf": f["dswrf"], "tcdc": f["tcdc"], "cs": f["cs"], "kc_nwp": f["kc"]}
        cov_stats[split] = round(float(np.isnan(f["dswrf"]).mean()), 4)
    print("share of missing GFS radiation by split:", cov_stats)

    flat = {s: d[f"X_{s}"].reshape(len(d[f"X_{s}"]), -1) for s in ["train", "val", "test"]}
    with_nwp = {s: np.hstack([flat[s], feats[s]["dswrf"], feats[s]["tcdc"], feats[s]["cs"],
                           feats[s]["kc_nwp"]])
                for s in ["train", "val", "test"]}
    trainer = TreeModelTrainer(config_path=args.model_config)
    preds, timing = {}, {}
    for name, X in [("lgbm_no_nwp", flat), ("lgbm_nwp", with_nwp)]:
        t0 = time.time()
        model = trainer.train(X["train"], d["y_train"], X["val"], d["y_val"], model_type="lightgbm")
        preds[name] = {s: model.predict(X[s]) for s in ["val", "test"]}
        timing[name] = round(time.time() - t0, 1)
        print(f"  {name}: trained in {timing[name]} s", flush=True)
    preds["gfs_only"] = {s: feats[s]["kc_nwp"] for s in ["val", "test"]}
    series = d["target_series"]
    per_day = int(pd.Timedelta("1D") / step)
    preds["SmartPersistence"] = {s: np.stack([series[t - per_day:t - per_day + H] for t in d["target_start_rows"][s]])
                                 for s in ["val", "test"]}

    y, mask = d["y_test"], d["daytime_mask"]
    rows = []
    for name, p in preds.items():
        m = MetricsFactory.compute_all(y, p["test"], mask, y_smart_persistence=preds["SmartPersistence"]["test"])
        m["Model"] = name
        rows.append(m)
    table = pd.DataFrame(rows).sort_values("MAE")
    print(table[["Model", "MAE", "RMSE", "R2", "Skill_Score_Daily"]].to_string(index=False))

    out = args.out_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"metrics_nwp_{args.horizon}.csv", index=False)
    np.savez_compressed(out / f"predictions_nwp_{args.horizon}.npz", y_val=d["y_val"], y_test=y,
                        daytime_mask_val=d["daytime_mask_val"], daytime_mask_test=mask,
                        **{f"{s}__{n}": p[s] for n, p in preds.items() for s in ["val", "test"]})
    (out / f"nwp_run_{args.horizon}.json").write_text(json.dumps(
        {"period_end": args.end, "n_train": int(len(d["y_train"])), "n_val": int(len(d["y_val"])),
         "n_test": int(len(y)), "missing_gfs_share": cov_stats, "train_seconds": timing}, indent=1))


if __name__ == "__main__":
    main()
