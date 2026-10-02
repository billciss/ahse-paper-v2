"""
reconstruct_available_power.py
==============================
Reconstructs the AVAILABLE (uncurtailed) power of the Yulara Desert Gardens site (DG),
the only curtailed site of the off-grid mini-grid, from the four uncurtailed neighbouring
sites (Service Station, Sails in the Desert, Laundry, Connellan Airport) that see the
same weather a few km away.

Training pool U ("uncurtailed with high confidence"): sun up and
    hour < 9  or  hour >= 16  or  total PV < TOTAL_UNCURTAILED_KW,
since observed network caps are rarely below ~450-500 kW (see quantify_curtailment.py).

Models (compared by leave-one-year-out on U, errors stratified by output level):
  ratio       : DG = k(solar-azimuth bin, season) * sum(neighbours), k = median ratio on U
  lgbm        : LightGBM on neighbour powers, solar geometry, GHI, clear-sky index, season
  kt_transfer : clear-sky-index transfer. Each neighbour gets a plane-of-array clear-sky
                envelope fitted on all its training rows (it is not curtailed); DG gets one
                fitted on its U rows only (both sides of noon identify its geometry).
                DG_available = (sum neighbours / sum neighbour envelopes) * DG envelope.
Physical check on all midday instants: curtailment only reduces output, so delivered DG
should rarely exceed the reconstructed available power.

The full series is produced OUT-OF-YEAR (each year predicted by a model trained on the
other years) to avoid in-sample optimism.

Usage (WSL):
    python scripts/reconstruct_available_power.py --downloads /mnt/c/Users/billc/Downloads \
        --out-csv data/processed/yulara_dg_available_15min.csv --metrics results/curtailment/reconstruction_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantify_curtailment import LAT, LON, ALT, TZ  # noqa: E402
from yulara_sites_analysis import SITES, read_power  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_engine.pv_physics import ClearSkyCalculator  # noqa: E402

NEIGHBOURS = ["ServiceStation", "SailsCombined", "Laundry", "ConnellanAirport"]
TOTAL_UNCURTAILED_KW = 350.0
FREQ = "15min"


def build_frame(downloads: Path) -> pd.DataFrame:
    cols = ["DesertGardens", "Total"] + NEIGHBOURS
    P = pd.DataFrame({k: read_power(downloads / SITES[k][0]) for k in cols})
    w = pd.read_csv(downloads / SITES["Total"][0], index_col=0,
                    usecols=["timestamp", "Global_Horizontal_Radiation"]).rename(
        columns={"Global_Horizontal_Radiation": "GHI"})
    w.index = pd.to_datetime(w.index)
    df = P.join(w).resample(FREQ).mean()
    df.index = df.index.tz_localize(TZ)
    loc = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
    sp = loc.get_solarposition(df.index)
    cs = loc.get_clearsky(df.index)
    df["zenith"] = sp["apparent_zenith"].to_numpy()
    df["azimuth"] = sp["azimuth"].to_numpy()
    df["ghi_cs"] = cs["ghi"].to_numpy()
    df["kc"] = (df["GHI"] / df["ghi_cs"].where(df["ghi_cs"] > 50)).clip(0, 1.5)
    df["neigh_sum"] = df[NEIGHBOURS].sum(axis=1, min_count=len(NEIGHBOURS))
    df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df["season"] = np.where(df.index.month.isin([4, 5, 6, 7, 8, 9]), "cool", "warm")
    df["az_bin"] = (df["azimuth"] // 10).astype("Int64")
    day = (df["zenith"] < 85) & (df["neigh_sum"] > 5)
    df = df[day].dropna(subset=["DesertGardens", "Total", "neigh_sum", "GHI"])
    hour = df.index.hour
    df["U"] = (hour < 9) | (hour >= 16) | (df["Total"] < TOTAL_UNCURTAILED_KW)
    return df


LGB_FEATURES = NEIGHBOURS + ["zenith", "azimuth", "GHI", "kc", "doy_sin", "doy_cos"]


def fit_ratio(train):
    train = train[train["U"]]
    k = (train["DesertGardens"] / train["neigh_sum"]).groupby([train["az_bin"], train["season"]]).median()
    k_az = (train["DesertGardens"] / train["neigh_sum"]).groupby(train["az_bin"]).median()
    k_all = float((train["DesertGardens"] / train["neigh_sum"]).median())
    return k, k_az, k_all


def predict_ratio(model, X):
    k, k_az, k_all = model
    idx = pd.MultiIndex.from_arrays([X["az_bin"], X["season"]])
    ratio = pd.Series(k.reindex(idx).to_numpy(), index=X.index)
    ratio = ratio.fillna(pd.Series(k_az.reindex(X["az_bin"]).to_numpy(), index=X.index)).fillna(k_all)
    return (ratio * X["neigh_sum"]).to_numpy()


def fit_lgbm(train):
    train = train[train["U"]]
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=42, verbose=-1)
    return m.fit(train[LGB_FEATURES], train["DesertGardens"])


def _envelope(site, rows, fit_rows, uncapped=False):
    calc = ClearSkyCalculator(LAT, LON, ALT, TZ)
    calc.fit_clear_sky_power(rows.index, rows[site], rows["GHI"], fit_rows)
    if uncapped:  # DG's own maximum is shaped by curtailment: do not cap its envelope
        calc.pcs_params["p_max"] = float("inf")
    return calc


def fit_kt_transfer(train):
    envs = {s: _envelope(s, train, np.ones(len(train), bool)) for s in NEIGHBOURS}
    envs["DesertGardens"] = _envelope("DesertGardens", train, train["U"].to_numpy(), uncapped=True)
    return envs


def predict_kt_transfer(envs, X):
    cs = {s: envs[s].calculate_clear_sky_power(X.index, X[s]).to_numpy() for s in envs}
    neigh_cs = sum(cs[s] for s in NEIGHBOURS)
    with np.errstate(divide="ignore", invalid="ignore"):
        kc = np.clip(X["neigh_sum"].to_numpy() / neigh_cs, 0, 1.5)
    return np.nan_to_num(kc * cs["DesertGardens"])


MODELS = {"ratio": (fit_ratio, predict_ratio),
          "lgbm": (fit_lgbm, lambda m, X: m.predict(X[LGB_FEATURES])),
          "kt_transfer": (fit_kt_transfer, predict_kt_transfer)}


def metrics(y, p, level):
    out = {}
    for name, sel in [("all", np.ones(len(y), bool))] + [
            (f"DG_{lo}-{hi}kW", (level >= lo) & (level < hi)) for lo, hi in [(0, 200), (200, 400), (400, 2000)]]:
        if sel.sum() < 50:
            continue
        e = p[sel] - y[sel]
        out[name] = {"n": int(sel.sum()), "MAE": round(float(np.mean(np.abs(e))), 2),
                     "bias": round(float(np.mean(e)), 2),
                     "nMAE_%": round(float(100 * np.mean(np.abs(e)) / np.mean(y[sel])), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path)
    ap.add_argument("--metrics", type=Path)
    args = ap.parse_args()

    df = build_frame(args.downloads)
    years = sorted(df.index.year.unique())
    print(f"daytime rows {len(df):,}; uncurtailed-pool U {int(df['U'].sum()):,} "
          f"({df['U'].mean():.1%}); years {years[0]}-{years[-1]}")

    report, oof = {}, {}
    for name, (fit, predict) in MODELS.items():
        y_all, p_all, lvl_all = [], [], []
        pred_full = pd.Series(np.nan, index=df.index)
        for yr in years:
            train = df[df.index.year != yr]
            test_all = df[df.index.year == yr]
            model = fit(train)
            pred_full.loc[test_all.index] = predict(model, test_all)
            test_u = test_all[test_all["U"]]
            y_all.append(test_u["DesertGardens"].to_numpy())
            p_all.append(pred_full.loc[test_u.index].to_numpy())
            lvl_all.append(test_u["DesertGardens"].to_numpy())
        y, p, lvl = map(np.concatenate, (y_all, p_all, lvl_all))
        mid = df[(~df["U"]) & (df.index.hour >= 11) & (df.index.hour < 14)]
        exceed = float((mid["DesertGardens"] > 1.05 * pred_full.loc[mid.index] + 5).mean())
        report[name] = {"leave_one_year_out_on_U": metrics(y, p, lvl),
                        "midday_share_delivered_above_available": round(exceed, 4)}
        oof[name] = pred_full
        print(f"\n[{name}] leave-one-year-out on U:")
        for k, v in report[name]["leave_one_year_out_on_U"].items():
            print(f"   {k:14s} {v}")
        print(f"   midday (11-14 h, outside U): delivered > 1.05*available + 5 kW in {exceed:.2%} of instants")

    # physically admissible models only (delivered rarely above available at midday),
    # then lowest error at high output
    ok = [m for m in MODELS if report[m]["midday_share_delivered_above_available"] < 0.10] or list(MODELS)
    best = min(ok, key=lambda m: report[m]["leave_one_year_out_on_U"]["DG_400-2000kW"]["MAE"])
    print(f"\nselected model (admissible, lowest MAE at high output): {best}")
    avail = oof[best].clip(lower=0)
    curt = (avail - df["DesertGardens"]).clip(lower=0)
    share_energy = float(curt.sum() / avail.sum())
    by_year = (curt.groupby(df.index.year).sum() / avail.groupby(df.index.year).sum()).round(3).to_dict()
    by_month = (curt.groupby(df.index.month).sum() / avail.groupby(df.index.month).sum()).round(3).to_dict()
    print(f"curtailed energy share of DG available energy: {share_energy:.1%}")
    print("   by year :", by_year)
    print("   by month:", by_month)
    report["selected"] = best
    report["curtailed_energy_share"] = round(share_energy, 4)
    report["curtailed_energy_share_by_year"] = {int(k): v for k, v in by_year.items()}
    report["curtailed_energy_share_by_month"] = {int(k): v for k, v in by_month.items()}

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"DG_delivered": df["DesertGardens"], "DG_available": avail,
                            "DG_curtailment": curt, "neighbours_sum": df["neigh_sum"],
                            "Total_delivered": df["Total"], "GHI": df["GHI"], "U": df["U"]})
        out.to_csv(args.out_csv, index_label="timestamp")
        print("series ->", args.out_csv)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
