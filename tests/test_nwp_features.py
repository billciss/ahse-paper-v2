"""NWP covariates must come only from runs published before the forecast issue time."""
import numpy as np
import pandas as pd

from src.data_engine.nwp_features import nwp_covariates


def _nwp():
    rows = []
    for init in pd.date_range("2020-01-01", periods=8, freq="6h", tz="UTC"):
        for f in range(3, 49, 3):
            valid = init + pd.Timedelta(hours=f)
            # encode the run and the valid time in the value to check the selection
            rows.append({"init": init, "valid": valid, "x": init.hour * 1000 + f})
    return pd.DataFrame(rows)


def test_uses_latest_run_published_before_issue_time():
    nwp = _nwp()
    # issued 2020-01-01 10:00 UTC: the 06Z run is published at 11:00 -> not usable; 00Z run is
    out = nwp_covariates(nwp, [pd.Timestamp("2020-01-01 10:00", tz="UTC")], 4, pd.Timedelta("1h"), ["x"])
    assert (out[0, :, 0] // 1000 == 0).all()
    # issued at 11:00 the 06Z run becomes usable
    out = nwp_covariates(nwp, [pd.Timestamp("2020-01-01 11:00", tz="UTC")], 4, pd.Timedelta("1h"), ["x"])
    assert (out[0, :, 0] // 1000 == 6).all()


def test_targets_map_to_the_three_hour_window_that_contains_them():
    nwp = _nwp()
    out = nwp_covariates(nwp, [pd.Timestamp("2020-01-01 10:00", tz="UTC")], 6, pd.Timedelta("1h"), ["x"])
    # 00Z run; hourly targets [11,12) .. [16,17) (midpoints 11:30 .. 16:30)
    # -> windows ending at 12, 15, 15, 15, 18, 18 h
    np.testing.assert_array_equal(out[0, :, 0] % 1000, [12, 15, 15, 15, 18, 18])


def test_no_run_available_gives_nan():
    out = nwp_covariates(_nwp(), [pd.Timestamp("2019-12-31 23:00", tz="UTC")], 3, pd.Timedelta("1h"), ["x"])
    assert np.isnan(out).all()


def test_three_hour_means_deaccumulate_six_hour_averages():
    from src.data_engine.nwp_features import three_hour_means
    init = pd.Timestamp("2020-01-01 00:00")
    # true 3-hour means: 100 over (0,3], 300 over (3,6] -> GFS stores avg3=100, avg6=200
    raw = pd.DataFrame({"init": [init, init], "fhour": [3, 6],
                        "win_start": [init, init],
                        "win_end": [init + pd.Timedelta("3h"), init + pd.Timedelta("6h")],
                        "value": [100.0, 200.0]})
    out = three_hour_means(raw)
    np.testing.assert_allclose(out["value_3h"], [100.0, 300.0])


def test_gfs_clearness_uses_clear_sky_of_the_same_window():
    import pvlib
    from src.data_engine.nwp_features import gfs_target_features
    loc = pvlib.location.Location(-25.24, 131.04, tz="Australia/Darwin", altitude=492)
    tidx = pd.date_range("2020-01-02", periods=72, freq="1h", tz="Australia/Darwin")
    # GFS radiation exactly equal to the clear-sky mean of each 3-hour window -> kc == 1 by day
    t5 = pd.date_range("2020-01-01", "2020-01-06", freq="5min", tz="UTC")
    cs3 = loc.get_clearsky(t5)["ghi"].resample("3h", label="right", closed="right").mean()
    rows = []
    for init in pd.date_range("2020-01-01", "2020-01-04", freq="6h", tz="UTC"):
        for f in range(3, 49, 3):
            v = init + pd.Timedelta(hours=f)
            rows.append({"init": init, "valid": v, "dswrf": cs3.get(v, np.nan), "tcdc": 0.0})
    gfs = pd.DataFrame(rows)
    f = gfs_target_features(gfs, tidx, np.array([24]), 24, pd.Timedelta("1h"), loc)
    day = f["cs"][0] > 200
    assert day.sum() > 5
    np.testing.assert_allclose(f["kc"][0][day], 1.0, atol=1e-6)
