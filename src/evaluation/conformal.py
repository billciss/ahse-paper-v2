"""
Conformal prediction intervals for multi-horizon k_t forecasts.

Scores are absolute residuals |y - p| of daytime targets. Calibration is Mondrian: one
quantile per (lead-time block, level of the predicted k_t), because errors grow with the
lead time and are much larger for intermediate k_t (broken clouds) than for clear or fully
overcast conditions. Quantiles use the finite-sample rank ceil((n + 1)(1 - alpha)).
The binning variable (`key`) defaults to the prediction itself; any quantity known at the
issue time can be used instead, e.g. the spread of the base models.

Two variants:
  split    fixed quantiles from the calibration scores;
  aci      adaptive conformal inference (Gibbs & Candes, 2021): the miscoverage level used
           each day is updated with the coverage errors of forecasts whose targets have all
           been observed (delay of `delay_days`), so the update never uses future values.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

DEFAULT_LEVEL_EDGES = (0.4, 0.8)     # predicted k_t bins: overcast / intermediate / clear


def _bins(pred: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.digitize(pred, edges)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile; +inf when there are too few scores."""
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    if k < 1:
        return 0.0
    return float(np.sort(scores)[k - 1])


class MondrianConformal:
    def __init__(self, blocks: Dict[str, Tuple[int, int]], level_edges: Sequence[float] = DEFAULT_LEVEL_EDGES,
                 min_count: int = 100):
        self.blocks = blocks
        self.edges = tuple(level_edges)
        self.min_count = min_count
        self.scores: Dict[Tuple[str, int], np.ndarray] = {}

    def _cells(self, key: np.ndarray):
        """Yield (block name, bin, boolean cell mask over key's shape)."""
        b = _bins(key, self.edges)
        steps = np.arange(key.shape[1])[None, :]
        for name, (a, z) in self.blocks.items():
            in_block = (steps >= a) & (steps < z)
            for k in range(len(self.edges) + 1):
                yield name, k, in_block & (b == k)

    def fit(self, y_cal: np.ndarray, p_cal: np.ndarray, mask_cal: np.ndarray,
            key: np.ndarray = None) -> "MondrianConformal":
        self.scores = {}
        res = np.abs(y_cal - p_cal)
        for name, k, cell in self._cells(p_cal if key is None else key):
            s = res[cell & mask_cal]
            if len(s) < self.min_count:          # too few points: pool the whole block
                a, z = self.blocks[name]
                s = res[:, a:z][mask_cal[:, a:z]]
            self.scores[(name, k)] = s
        return self

    def radius(self, pred: np.ndarray, alpha, key: np.ndarray = None) -> np.ndarray:
        """Interval half-width for every target; alpha may be a scalar or one value per row."""
        alpha = np.broadcast_to(np.asarray(alpha, dtype=float).reshape(-1, 1), pred.shape)
        out = np.full(pred.shape, np.nan)
        for name, k, cell in self._cells(pred if key is None else key):
            s = np.sort(self.scores[(name, k)])
            n = len(s)
            for a_val in np.unique(alpha[cell]):
                sel = cell & (alpha == a_val)
                out[sel] = conformal_quantile(s, a_val) if n else np.inf
        return out

    def interval(self, pred: np.ndarray, alpha, lower: float = 0.0,
                 key: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        r = self.radius(pred, alpha, key)
        return np.maximum(pred - r, lower), pred + r


def adaptive_alphas(cover_fn, mask: np.ndarray, day_ids: np.ndarray, alpha: float,
                    gamma: float = 0.02, delay_days: int = 2, bounds: Tuple[float, float] = (0.01, 0.5)):
    """Adaptive conformal inference over forecast days (rows grouped by issue day).

    cover_fn(rows, alpha_t) returns the boolean coverage of the forecasts in `rows` when their
    intervals use miscoverage level alpha_t. The level of day d is updated only with the
    coverage errors of days <= d - delay_days, whose targets are all observed by then.
    Returns the per-row alpha used and the boolean coverage matrix.
    """
    days = np.unique(day_ids)
    alpha_t = alpha
    alpha_rows = np.empty(len(day_ids))
    cov = np.zeros(mask.shape, dtype=bool)
    pending = []                               # (day, error) waiting for their outcomes
    for d in days:
        # outcomes of forecasts issued delay_days earlier are now fully observed
        ready = [e for (dd, e) in pending if dd <= d - delay_days]
        pending = [(dd, e) for (dd, e) in pending if dd > d - delay_days]
        for e in ready:
            alpha_t = float(np.clip(alpha_t + gamma * (alpha - e), *bounds))
        rows = day_ids == d
        alpha_rows[rows] = alpha_t
        cov_d = cover_fn(rows, alpha_t)
        cov[rows] = cov_d
        m = mask[rows]
        if m.any():
            pending.append((d, float(1 - cov_d[m].mean())))
    return alpha_rows, cov


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray, alpha: float) -> dict:
    """Coverage, mean width and interval (Winkler) score on masked targets."""
    yy, l, h = y[mask], lo[mask], hi[mask]
    width = h - l
    penalty = (2 / alpha) * ((l - yy) * (yy < l) + (yy - h) * (yy > h))
    return {"coverage": float(((yy >= l) & (yy <= h)).mean()), "width": float(width.mean()),
            "interval_score": float((width + penalty).mean()), "n": int(mask.sum())}
