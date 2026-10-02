"""Conformal intervals: finite-sample coverage on exchangeable data, Mondrian cells, ACI."""
import numpy as np

from src.evaluation.conformal import MondrianConformal, adaptive_alphas, conformal_quantile, interval_metrics

BLOCKS = {"early": (0, 4), "late": (4, 8)}


def _data(n, rng, late_scale=3.0):
    p = rng.uniform(0, 1.2, (n, 8))
    scale = np.where(np.arange(8) < 4, 0.05, 0.05 * late_scale)
    y = p + rng.normal(0, 1, (n, 8)) * scale
    return y, p, np.ones_like(y, dtype=bool)


def test_quantile_rank():
    s = np.arange(1, 10, dtype=float)               # n = 9
    assert conformal_quantile(s, 0.1) == 9.0        # ceil(10 * 0.9) = 9
    assert conformal_quantile(s, 0.05) == np.inf    # ceil(10 * 0.95) = 10 > n


def test_split_coverage_and_block_widths():
    rng = np.random.default_rng(0)
    cp = MondrianConformal(BLOCKS).fit(*_data(4000, rng))
    y, p, m = _data(4000, rng)
    lo, hi = cp.interval(p, 0.1, lower=-np.inf)
    for (a, b) in BLOCKS.values():
        mm = np.zeros_like(m)
        mm[:, a:b] = True
        assert abs(interval_metrics(y, lo, hi, mm, 0.1)["coverage"] - 0.9) < 0.02
    assert (hi - lo)[:, 4:].mean() > 2 * (hi - lo)[:, :4].mean()


def test_aci_recovers_coverage_under_shift():
    rng = np.random.default_rng(1)
    cp = MondrianConformal(BLOCKS).fit(*_data(2000, rng))
    y, p, m = _data(3000, rng, late_scale=6.0)        # errors grew after calibration
    days = np.repeat(np.arange(300), 10)

    def cover(rows, a):
        lo, hi = cp.interval(p[rows], a, lower=-np.inf)
        return (y[rows] >= lo) & (y[rows] <= hi)

    split_cov = cover(np.ones(len(y), bool), 0.1)[m].mean()
    _, cov = adaptive_alphas(cover, m, days, 0.1, gamma=0.05)
    assert split_cov < 0.8
    assert cov[days >= 150].mean() > split_cov + 0.05
