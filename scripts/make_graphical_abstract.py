"""
make_graphical_abstract.py
==========================
Graphical abstract following the Elsevier guidelines: 5:2 ratio, 3321 x 1329 px at 300 dpi
(minimum 1328 x 531), Arial, left-to-right reading, no title, no empty margins. The numbers are
read from results/power_summary (test MAE of AHSE in kW, with and without GFS) and the conformal
tables, so the figure stays consistent with the paper.

Outputs: paper/figures/graphical_abstract.{pdf,tif,png}

Usage (WSL):  python scripts/make_graphical_abstract.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

for f in ("arial.ttf", "arialbd.ttf"):           # Windows Arial, as required by Elsevier
    p = Path("/mnt/c/Windows/Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

BLUE, ORANGE, GREEN, GREY, INK = "#1f4e8c", "#e08a1e", "#2e7d4f", "#8a8f98", "#1d2433"
PANEL = {"data": "#eaf2fb", "method": "#fdf6e3", "results": "#eef6f0"}


def mae(site, arm):
    t = pd.read_csv(ROOT / f"results/power_summary/{site}_{arm}_power_1h.csv", index_col=0)
    return float(t.loc["AHSE", "MAE_kW_mean"])


def coverage_range():
    covs = []
    for run in ("results/yulara_neighbours_gfs", "results/nist_gfs", "results/yulara_neighbours"):
        t = pd.read_csv(ROOT / run / "tables/conformal_1h.csv")
        t = t[(t.nominal == 0.9) & (t.group == "all") & t.method.isin(["split_mondrian", "aci_mondrian"])]
        covs += list(t.coverage)
    return min(covs), max(covs)


def box(ax, x, y, w, h, fc, ec="none", lw=1.5, r=0.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def text(ax, x, y, s, size, color=INK, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>,head_length=0.55,head_width=0.4",
                                 mutation_scale=18, lw=3.5, color=BLUE))


def pv_icon(ax, x, y, s=1.0):
    ax.add_patch(Circle((x - 1.0 * s, y + 1.15 * s), 0.42 * s, color=ORANGE))
    for k in range(8):
        import math
        a = k * math.pi / 4
        ax.plot([x - 1.0 * s + 0.55 * s * math.cos(a), x - 1.0 * s + 0.75 * s * math.cos(a)],
                [y + 1.15 * s + 0.55 * s * math.sin(a), y + 1.15 * s + 0.75 * s * math.sin(a)],
                color=ORANGE, lw=2.5, solid_capstyle="round")
    p = [(x - 0.4 * s, y - 0.2 * s), (x + 1.6 * s, y - 0.2 * s), (x + 1.95 * s, y + 0.75 * s), (x - 0.05 * s, y + 0.75 * s)]
    ax.add_patch(Polygon(p, closed=True, fc="#9cc3ea", ec=BLUE, lw=2))
    for t in (1 / 3, 2 / 3):
        ax.plot([p[0][0] + t * (p[1][0] - p[0][0]), p[3][0] + t * (p[2][0] - p[3][0])],
                [p[0][1], p[3][1]], color=BLUE, lw=1.2)
    ax.plot([p[0][0] + (p[3][0] - p[0][0]) / 2, p[1][0] + (p[2][0] - p[1][0]) / 2],
            [(p[0][1] + p[3][1]) / 2] * 2, color=BLUE, lw=1.2)
    ax.plot([x + 0.8 * s, x + 0.8 * s], [y - 0.2 * s, y - 0.75 * s], color=INK, lw=3)


def cloud_icon(ax, x, y, s=1.0):
    for dx, dy, w, h in ((0, 0, 1.5, 0.75), (0.45, 0.3, 1.1, 0.85), (-0.4, 0.15, 0.9, 0.6)):
        ax.add_patch(Ellipse((x + dx * s, y + dy * s), w * s, h * s, fc="white", ec=BLUE, lw=2))
    ax.add_patch(Ellipse((x, y), 1.45 * s, 0.7 * s, fc="white", ec="none"))


def main():
    w_in, h_in, dpi = 3321 / 300, 1329 / 300, 300
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---------------- panel 1: data
    box(ax, 0.15, 0.2, 7.3, 9.6, PANEL["data"], ec=BLUE)
    text(ax, 3.8, 9.1, "Physical PV data", 17, BLUE, "bold")
    pv_icon(ax, 1.75, 6.35, 0.85)
    text(ax, 5.25, 7.25, "Yulara  768 kW", 14, weight="bold")
    text(ax, 5.25, 6.65, "off-grid, desert", 12, GREY)
    text(ax, 5.25, 5.75, "NIST  271 kW", 14, weight="bold")
    text(ax, 5.25, 5.15, "grid-connected, humid", 12, GREY)
    box(ax, 0.6, 2.85, 6.4, 1.45, "white", ec="#c23b22", lw=2)
    text(ax, 3.8, 3.8, "Curtailed series removed", 13.5, "#c23b22", "bold")
    text(ax, 3.8, 3.25, "≈50% of its energy curtailed", 12, "#c23b22")
    cloud_icon(ax, 1.15, 1.4, 0.7)
    text(ax, 4.6, 1.65, "Archived GFS forecasts", 13, weight="bold")
    text(ax, 4.6, 1.05, "only runs published in time", 11.5, GREY)

    arrow(ax, 7.55, 8.5, 5.0)

    # ---------------- panel 2: honest selection
    box(ax, 8.6, 0.2, 7.8, 9.6, PANEL["method"], ec=ORANGE)
    text(ax, 12.5, 9.1, "Honest selection (AHSE)", 17, "#9a5b05", "bold")
    models = ["LightGBM", "XGBoost", "PatchTST", "N-HiTS", "LSTM", "GRU"]
    for k, m in enumerate(models):
        cx, cy = 10.0 + (k % 3) * 2.5, 7.85 - (k // 3) * 0.95
        box(ax, cx - 1.1, cy - 0.34, 2.2, 0.68, "white", ec=ORANGE, lw=1.5, r=0.3)
        text(ax, cx, cy, m, 12)
    text(ax, 12.5, 5.95, "+ GFS inputs for tree models", 12, GREY)
    box(ax, 9.1, 2.5, 6.8, 2.9, "white", ec=GREEN, lw=2.2)
    text(ax, 12.5, 4.85, "per lead-time block: 1 h, 1–6 h, 6–24 h", 12)
    text(ax, 12.5, 4.1, "combine models only if the", 14, GREEN, "bold")
    text(ax, 12.5, 3.5, "bootstrap gain is > 0", 14, GREEN, "bold")
    text(ax, 12.5, 2.9, "otherwise: best single model", 12, GREY)
    text(ax, 12.5, 1.65, "Decided on validation data only", 13, weight="bold")
    text(ax, 12.5, 1.0, "+ out-of-fold conformal 90% intervals", 12, GREY)

    arrow(ax, 16.5, 17.45, 5.0)

    # ---------------- panel 3: results
    box(ax, 17.55, 0.2, 7.3, 9.6, PANEL["results"], ec=GREEN)
    text(ax, 21.2, 9.1, "Test results, 1–24 h ahead", 17, GREEN, "bold")
    vals = {"Yulara": (mae("yulara", "no_gfs"), mae("yulara", "gfs")),
            "NIST": (mae("nist", "no_gfs"), mae("nist", "gfs"))}
    ins = fig.add_axes([0.775, 0.36, 0.2, 0.45])
    xs = [0, 1, 2.6, 3.6]
    hs = [vals["Yulara"][0], vals["Yulara"][1], vals["NIST"][0], vals["NIST"][1]]
    bars = ins.bar(xs, hs, width=0.85, color=[GREY, BLUE, GREY, BLUE])
    for x, h in zip(xs, hs):
        ins.text(x, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=11, color=INK)
    for x0, (a, b) in ((0.5, vals["Yulara"]), (3.1, vals["NIST"])):
        ins.text(x0, max(a, b) + 5.5, f"−{100 * (1 - b / a):.0f}%", ha="center", fontsize=16,
                 color=BLUE, weight="bold")
    ins.set_xticks([0.5, 3.1])
    ins.set_xticklabels(["Yulara", "NIST"], fontsize=13, weight="bold")
    ins.set_ylabel("AHSE MAE (kW)", fontsize=11.5)
    ins.set_ylim(0, 47)
    ins.set_xlim(-0.6, 4.2)
    ins.tick_params(axis="y", labelsize=10.5)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)
    ins.set_facecolor(PANEL["results"])
    ins.legend([bars[0], bars[1]], ["without GFS", "with GFS"], fontsize=11, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, handlelength=1.2)
    lo, hi = coverage_range()
    text(ax, 21.2, 1.55, f"90% intervals: coverage {lo:.2f}–{hi:.2f}", 13, weight="bold")
    text(ax, 21.2, 0.9, "AHSE never significantly worse than best model", 10.5, GREY)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "graphical_abstract.pdf")
    fig.savefig(OUT / "graphical_abstract.png", dpi=dpi)
    fig.savefig(OUT / "graphical_abstract.tif", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    im = Image.open(OUT / "graphical_abstract.tif")
    print("written", OUT / "graphical_abstract.*", im.size, im.info.get("dpi"))


if __name__ == "__main__":
    main()
