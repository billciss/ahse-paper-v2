"""
graphical_abstract_values.py
============================
Numbers shown in the graphical abstract, read from results/power_summary (test MAE of AHSE in kW
with and without GFS) and the conformal tables; written to paper/figures/graphical_abstract_values.json
for scripts/make_graphical_abstract_pptx.js.

Usage (WSL, repository root):  python scripts/graphical_abstract_values.py
"""
import json, pandas as pd
v = {}
for site in ("yulara", "nist"):
    for arm in ("no_gfs", "gfs"):
        t = pd.read_csv(f"results/power_summary/{site}_{arm}_power_1h.csv", index_col=0)
        v[f"{site}_{arm}_exact"] = float(t.loc["AHSE", "MAE_kW_mean"])
        v[f"{site}_{arm}"] = round(v[f"{site}_{arm}_exact"], 1)
covs = []
for run in ("results/yulara_neighbours_gfs", "results/nist_gfs", "results/yulara_neighbours"):
    t = pd.read_csv(run + "/tables/conformal_1h.csv")
    t = t[(t.nominal == 0.9) & (t.group == "all") & t.method.isin(["split_mondrian", "aci_mondrian"])]
    covs += list(t.coverage)
v["cov_lo"], v["cov_hi"] = round(min(covs), 2), round(max(covs), 2)
v["red_yulara"] = round(100 * (1 - v["yulara_gfs_exact"] / v["yulara_no_gfs_exact"]))
v["red_nist"] = round(100 * (1 - v["nist_gfs_exact"] / v["nist_no_gfs_exact"]))
json.dump(v, open("paper/figures/graphical_abstract_values.json", "w"), indent=1)
print(v)
