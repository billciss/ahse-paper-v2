"""Long gaps must not be silently interpolated into training/evaluation windows."""
import numpy as np
import pandas as pd
import yaml

from src.data_engine.preprocessor import MultiHorizonPreprocessor


def _config(tmp_path, policy):
    cfg = {"location": {"latitude": -25.24, "longitude": 131.04, "altitude": 492, "timezone": "Australia/Darwin"},
           "preprocessing": {"use_clear_sky_index": False, "scaler_type": "standard",
                             "gap_policy": policy, "max_interpolation_gap": "1h"}}
    path = tmp_path / f"cfg_{policy}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _frame(n=2000, long_gap=(800, 900), short_gap=(300, 302)):
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Active_Power": rng.uniform(size=n), "GHI": rng.uniform(size=n),
                       "Temperature": rng.uniform(size=n)}, index=idx)
    df.iloc[long_gap[0]:long_gap[1], 0] = np.nan      # 25 h target gap
    df.iloc[short_gap[0]:short_gap[1], 1] = np.nan    # 30 min feature gap
    return df


def test_exclude_policy_drops_windows_touching_long_gaps(tmp_path):
    pre = MultiHorizonPreprocessor(config_path=_config(tmp_path, "exclude"))
    out = pre.prepare_multi_horizon_data(_frame(), horizon="15min", lookback_window=8, forecast_horizon=4,
                                         train_ratio=0.6, val_ratio=0.2)
    # training split = rows 0..1199; windows are 12 rows long; the gap covers rows 800..899,
    # so windows starting in 789..899 (111 of them) must be removed; the short gap is kept
    n_windows = 1200 - 12 + 1
    assert len(out["X_train"]) == n_windows - 111
    assert np.isfinite(out["X_train"]).all() and np.isfinite(out["y_train"]).all()


def test_interpolate_policy_keeps_legacy_behaviour(tmp_path):
    pre = MultiHorizonPreprocessor(config_path=_config(tmp_path, "interpolate"))
    out = pre.prepare_multi_horizon_data(_frame(), horizon="15min", lookback_window=8, forecast_horizon=4,
                                         train_ratio=0.6, val_ratio=0.2)
    assert len(out["X_train"]) == 1200 - 12 + 1


def test_target_start_rows_align_with_sequences(tmp_path):
    pre = MultiHorizonPreprocessor(config_path=_config(tmp_path, "exclude"))
    out = pre.prepare_multi_horizon_data(_frame(), horizon="15min", lookback_window=8, forecast_horizon=4,
                                         train_ratio=0.6, val_ratio=0.2)
    series = out["target_series"]
    for split in ["train", "val", "test"]:
        starts, y = out["target_start_rows"][split], out[f"y_{split}"]
        assert len(starts) == len(y)
        rebuilt = np.stack([series[s:s + 4] for s in starts])
        np.testing.assert_allclose(rebuilt, y)
