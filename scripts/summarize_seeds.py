"""
summarize_seeds.py
==================
Aggregates metrics_global_<h>.csv over several seed runs (mean, standard deviation, min, max
per model) to check that differences between methods exceed seed-to-seed variability.

Usage:
    python scripts/summarize_seeds.py --runs results/yulara_neighbours results/seeds/seed_*/yulara_neighbours \
        --horizon 1h --out results/seeds/summary_1h.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--horizon", default="1h")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    frames = []
    for run in args.runs:
        f = run / "tables" / f"metrics_global_{args.horizon}.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(run=str(run)))
        else:
            print("missing:", f)
    df = pd.concat(frames, ignore_index=True)
    metrics = [c for c in ["MAE", "RMSE", "R2", "Skill_Score_Daily"] if c in df.columns]
    summary = df.groupby("Model")[metrics].agg(["mean", "std", "min", "max"]).round(5)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.sort_values("MAE_mean")
    summary.insert(0, "n_runs", df.groupby("Model").size())
    print(summary.to_string())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out)


if __name__ == "__main__":
    main()
