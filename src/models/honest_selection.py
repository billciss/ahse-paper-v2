"""
Honest per-horizon-block ensemble selection (AHSE, block version).

For each lead-time block, candidates are every single base model and the averages of the
top-k models ranked on that block's validation MAE. The reference is the best single model
over the whole horizon. A candidate replaces the reference on a block only if the lower
confidence bound (default 95%) of its validation MAE gain is positive, the gain being
bootstrapped over whole days to respect temporal dependence. All errors are computed on
daytime targets only, exactly as in the final evaluation. The test set is never used.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Candidate = Tuple[str, ...]          # a single model is a 1-tuple, an average a k-tuple


def default_blocks(horizon_steps: int, steps_per_hour: int) -> Dict[str, Tuple[int, int]]:
    """First hour, 1-6 h and the rest of the horizon."""
    first, six = steps_per_hour, min(6 * steps_per_hour, horizon_steps)
    return {"first_hour": (0, first), "1_to_6h": (first, six), "6h_to_end": (six, horizon_steps)}


def _masked_mae(y, p, m):
    return float(np.abs(y - p)[m].mean()) if m.any() else np.inf


class HonestBlockSelector:
    def __init__(self, blocks: Dict[str, Tuple[int, int]], top_k: Sequence[int] = (2, 3, 4),
                 exclude: Sequence[str] = ("Persistence", "AHSE", "AHSE_global"),
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
        self.blocks = blocks
        self.top_k = tuple(top_k)
        self.exclude = set(exclude)
        self.n_boot = n_boot
        self.alpha = alpha
        self.seed = seed
        self.reference: Optional[str] = None
        self.decisions: Dict[str, dict] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _combine(preds: Dict[str, np.ndarray], cand: Candidate) -> np.ndarray:
        return np.mean([preds[m] for m in cand], axis=0)

    def _lcb_gain(self, y, p_ref, p_new, m, days, rng) -> float:
        err_ref = np.where(m, np.abs(y - p_ref), 0).sum(axis=1)
        err_new = np.where(m, np.abs(y - p_new), 0).sum(axis=1)
        cnt = m.sum(axis=1)
        uniq, inv = np.unique(days, return_inverse=True)
        s_ref, s_new = np.bincount(inv, err_ref), np.bincount(inv, err_new)
        s_cnt = np.bincount(inv, cnt)
        idx = rng.integers(0, len(uniq), (self.n_boot, len(uniq)))
        gains = (s_ref[idx].sum(1) - s_new[idx].sum(1)) / np.maximum(s_cnt[idx].sum(1), 1)
        return float(np.percentile(gains, 100 * self.alpha))

    # ------------------------------------------------------------------ API
    def fit(self, preds_val: Dict[str, np.ndarray], y_val: np.ndarray, mask_val: np.ndarray,
            day_ids: np.ndarray) -> "HonestBlockSelector":
        rng = np.random.default_rng(self.seed)
        models: List[str] = sorted(m for m in preds_val if m not in self.exclude)
        self.reference = min(models, key=lambda m: _masked_mae(y_val, preds_val[m], mask_val))
        self.decisions = {}
        for name, (a, b) in self.blocks.items():
            sl = slice(a, b)
            yb, mb = y_val[:, sl], mask_val[:, sl]
            rank = sorted(models, key=lambda m: _masked_mae(yb, preds_val[m][:, sl], mb))
            cands: List[Candidate] = [(m,) for m in models] + \
                [tuple(rank[:k]) for k in self.top_k if k <= len(rank)]
            mae = {c: _masked_mae(yb, self._combine(preds_val, c)[:, sl], mb) for c in cands}
            ref = (self.reference,)
            lcb = {c: self._lcb_gain(yb, preds_val[self.reference][:, sl],
                                     self._combine(preds_val, c)[:, sl], mb, day_ids, rng)
                   for c in cands if c != ref}
            eligible = [c for c, g in lcb.items() if g > 0]
            choice = min(eligible, key=mae.get) if eligible else ref
            self.decisions[name] = {"steps": (a, b), "chosen": list(choice),
                                    "val_mae_chosen": mae[choice], "val_mae_reference": mae[ref],
                                    "lcb_gain": lcb.get(choice, 0.0)}
        return self

    def predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("HonestBlockSelector.predict() called before fit()")
        out = np.array(preds[self.reference], dtype=float, copy=True)
        for d in self.decisions.values():
            a, b = d["steps"]
            out[:, a:b] = self._combine(preds, tuple(d["chosen"]))[:, a:b]
        return out
