"""Interleaved validation: no window mixes training and validation rows, test stays last."""
import numpy as np

from src.data_engine.preprocessor import MultiHorizonPreprocessor


def test_interleaved_split_embargo_and_test_block():
    pre = MultiHorizonPreprocessor(config_path="configs/data_config_yulara_neighbours.yaml")
    pre.gap_policy = "exclude"
    n, L, H, val_end, block = 24 * 200, 24, 24, 24 * 160, 24 * 7
    pre.long_gap_rows = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(n, 3)), rng.uniform(size=n)
    is_val = np.zeros(n, dtype=bool)
    is_val[:val_end] = (np.arange(val_end) // block) % 4 == 3
    Xtr, ytr, Xva, yva, Xte, yte, starts = pre._interleaved_split(X, y, is_val, val_end, L, H)
    for s, want in ((starts[0], False), (starts[1], True)):
        rows = (s - L)[:, None] + np.arange(L + H)[None, :]          # every row of each window
        assert (is_val[rows] == want).all()
    assert starts[2].min() >= val_end + L
    assert len(Xva) > 0 and len(Xtr) > 3 * len(Xva) * 0.8
    # validation windows come from every block of the cycle, i.e. spread over the whole span
    assert np.ptp(starts[1]) > 0.6 * val_end
    # targets are the right rows
    np.testing.assert_allclose(yva[:, 0], y[starts[1]])
