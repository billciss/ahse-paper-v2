"""
make_paper_figures.py
=====================
Figures of the revised manuscript (paper/figures/):
  fig_curtailment.pdf   output of each Yulara site relative to the others under high irradiance,
                        by hour (results/curtailment/site_to_reference_ratio_by_hour.csv)
  fig_mae_leadtime.pdf  test MAE in kW by lead time, with and without GFS, Yulara and NIST
                        (cached predictions x clear-sky power envelope, seed 42)

Usage (WSL):  python scripts/make_paper_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_engine.preprocessor import load_and_prepare_multi_horizon  # noqa: E402

OUT = ROOT / "paper" / "figures"
SITES = {
    "Yulara (uncurtailed sites, 768 kW)": ("data/processed/yulara_neighbours_1h.csv",
                                            "configs/data_config_yulara_neighbours.yaml",
                                            "results/yulara_neighbours", "results/yulara_neighbours_gfs"),
    "NIST Ground array (271 kW)": ("data/processed/nist_pvdaq_1h.csv", "configs/data_config_nist.yaml",
                                   "results/nist", "results/nist_gfs"),
}


def curtailment():
    r = pd.read_csv(ROOT / "results/curtailment/site_to_reference_ratio_by_hour.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    names = {"DesertGardens": "Desert Gardens (1058 kW)", "ServiceStation": "Service Station",
             "SailsCombined": "Sails in the Desert", "Laundry": "Laundry", "ConnellanAirport": "Connellan Airport"}
    colors = {"DesertGardens": "C3", "ServiceStation": "C0", "SailsCombined": "C2", "Laundry": "C4",
              "ConnellanAirport": "C1"}
    for c in r.columns:
        ax.plot(r.index, r[c], marker="o", ms=3, lw=2.2 if c == "DesertGardens" else 1.2,
                color=colors.get(c), label=names.get(c, c))
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Output relative to the other sites\n(normalised to 08-09 h)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    fig.set_size_inches(6.4, 3.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_curtailment.pdf")
    fig.savefig(OUT / "fig_curtailment.png", dpi=200)


def mae_by_lead():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)
    for ax, (title, (data, cfg, run_a, run_b)) in zip(axes, SITES.items()):
        d = load_and_prepare_multi_horizon(str(ROOT / data), config_path=str(ROOT / cfg), horizon="1h")
        pcs = np.asarray(d["clear_sky_power"], dtype=float)
        target = yaml.safe_load(open(ROOT / cfg))["preprocessing"]["target_column"]
        power = pd.read_csv(ROOT / data, index_col=0, parse_dates=True)[target].to_numpy()
        for run, style in ((run_a, "--"), (run_b, "-")):
            z = np.load(ROOT / run / "tables" / "predictions_1h.npz")
            starts = z["target_start_rows_test"]
            rows = starts[:, None] + np.arange(z["y_test"].shape[1])[None, :]
            mask = z["daytime_mask_test"].astype(bool) & np.isfinite(power[rows])
            for m, color in (("AHSE", "C0"), ("XGBoost", "C1"), ("SmartPersistence", "0.4")):
                if run == run_b and m == "SmartPersistence":
                    continue
                err = np.abs(z[f"test__{m}"] * pcs[rows] - power[rows])
                mae = [err[:, h][mask[:, h]].mean() for h in range(err.shape[1])]
                lab = {"SmartPersistence": "day-ahead persistence"}.get(m, m)
                lab += "" if m == "SmartPersistence" else (" + GFS" if run == run_b else " (no GFS)")
                ax.plot(np.arange(1, len(mae) + 1), mae, style, color=color, lw=1.5, label=lab)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Lead time (h)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test MAE (kW)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mae_leadtime.pdf")
    fig.savefig(OUT / "fig_mae_leadtime.png", dpi=200)


def pipeline():
    """Schematic of the revised AHSE pipeline (graphical abstract)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    boxes = [
        (0.1, "Data", "Yulara: 4 uncurtailed\nsites (768 kW)\nNIST Ground (271 kW)\ncurtailment check", "#dbeafe"),
        (2.1, "Target", "k_t = P / P_cs\nP_cs: clear-sky POA\nenvelope fitted on\ntraining clear sky", "#dcfce7"),
        (4.1, "Members", "LightGBM, XGBoost\n(+ GFS published\nbefore issue time)\nPatchTST, N-HiTS,\nLSTM, GRU, persistence", "#fef3c7"),
        (6.1, "AHSE selection", "per lead-time block\nsingle or top-k average\naccepted if bootstrap\nLCB of gain > 0\n(validation only)", "#fde68a"),
        (8.1, "Output", "power forecast\n1-24 h ahead\n+ out-of-fold\nconformal intervals\n(Mondrian / ACI)", "#ede9fe"),
    ]
    for x, title, body, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.8, 2.9, boxstyle="round,pad=0.05", fc=color, ec="0.3"))
        ax.text(x + 0.9, 2.95, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + 0.9, 1.7, body, ha="center", va="center", fontsize=8)
    for x in (1.95, 3.95, 5.95, 7.95):
        ax.annotate("", xy=(x + 0.12, 1.8), xytext=(x - 0.02, 1.8),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="0.3"))
    ax.text(5.0, 0.08, "Chronological split 60/20/20 - evaluation in kW on daytime targets - "
            "day-level bootstrap and Diebold-Mariano tests - 5 seeds at Yulara", ha="center", fontsize=8, color="0.25")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.pdf")
    fig.savefig(OUT / "fig_pipeline.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    curtailment()
    mae_by_lead()
    print("figures written to", OUT)
