"""Tests for the clear-sky PV power envelope used to define the k_t target."""
import numpy as np
import pandas as pd
import pvlib
import pytest

from src.data_engine.pv_physics import ClearSkyCalculator

LAT, LON, ALT, TZ = -25.24, 131.04, 492, "Australia/Darwin"
TRUE_TILT, TRUE_AZ, TRUE_C, CAP = 20.0, 0.0, 0.3, 280.0   # north-facing array, kW per W/m2, inverter cap


def _synthetic_site(days=365, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2021-01-01", periods=days * 96, freq="15min", tz=TZ)
    loc = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
    cs = loc.get_clearsky(times, model="ineichen")
    sp = loc.get_solarposition(times)
    poa = pvlib.irradiance.get_total_irradiance(
        TRUE_TILT, TRUE_AZ, sp["apparent_zenith"], sp["azimuth"],
        cs["dni"], cs["ghi"], cs["dhi"])["poa_global"].fillna(0)
    p_clear = np.minimum(TRUE_C * poa, CAP)

    # 60 % clear days, 40 % cloudy days with a smooth random attenuation
    clear_day = rng.uniform(size=days) < 0.6
    att = np.ones(len(times))
    for d in np.flatnonzero(~clear_day):
        s = slice(d * 96, (d + 1) * 96)
        att[s] = np.clip(0.55 + 0.35 * np.sin(np.linspace(0, rng.uniform(3, 12), 96)), 0.1, 1)
    att *= rng.normal(1, 0.01, len(times))
    power = pd.Series(p_clear.to_numpy() * att, index=times).clip(lower=0)
    ghi = pd.Series(cs["ghi"].to_numpy() * att, index=times).clip(lower=0)
    return times, power, ghi, np.repeat(clear_day, 96)


@pytest.fixture(scope="module")
def site():
    return _synthetic_site()


def _calc(**kw):
    return ClearSkyCalculator(LAT, LON, ALT, TZ, **kw)


def test_fit_recovers_array_geometry_and_scale(site):
    times, power, ghi, _ = site
    calc = _calc()
    fit_mask = np.ones(len(times), bool)
    params = calc.fit_clear_sky_power(times, power, ghi, fit_mask)
    assert abs(params["tilt"] - TRUE_TILT) <= 5
    assert abs(((params["azimuth"] - TRUE_AZ + 180) % 360) - 180) <= 30
    assert params["scale"] == pytest.approx(TRUE_C, rel=0.05)


def test_kt_is_not_saturated_and_near_one_on_clear_days(site):
    times, power, ghi, clear = site
    calc = _calc()
    kt, p_cs = calc.calculate_kt(times, power, epsilon=1.0, kt_max=1.5,
                                 measured_ghi=ghi, fit_mask=np.ones(len(times), bool))
    day = (p_cs > 0.2 * CAP).to_numpy()
    assert np.mean(np.isclose(kt[day], 1.5)) < 0.01
    assert np.median(kt[day & clear]) == pytest.approx(1.0, abs=0.05)
    # the old single-ratio envelope saturated in winter; the plane-of-array one must not
    winter = day & clear & np.asarray(times.month.isin([6, 7]))
    assert np.median(kt[winter]) == pytest.approx(1.0, abs=0.07)


def test_fit_uses_training_rows_only(site):
    times, power, ghi, _ = site
    fit_mask = np.arange(len(times)) < int(0.6 * len(times))
    p1 = _calc().fit_clear_sky_power(times, power, ghi, fit_mask)
    tampered = power.copy()
    tampered[~fit_mask] *= 3.0
    p2 = _calc().fit_clear_sky_power(times, tampered, ghi, fit_mask)
    assert p1 == p2


def test_known_geometry_is_used_as_given(site):
    times, power, ghi, _ = site
    params = _calc(array_tilt=TRUE_TILT, array_azimuth=TRUE_AZ).fit_clear_sky_power(
        times, power, ghi, np.ones(len(times), bool))
    assert (params["tilt"], params["azimuth"]) == (TRUE_TILT, TRUE_AZ)
    assert params["scale"] == pytest.approx(TRUE_C, rel=0.03)


def test_too_few_clear_samples_raises(site):
    times, power, ghi, _ = site
    with pytest.raises(ValueError):
        _calc().fit_clear_sky_power(times, power, ghi * 0.3, np.ones(len(times), bool))


def test_missing_measured_ghi_raises(site):
    times, power, _, _ = site
    with pytest.raises(ValueError):
        _calc().calculate_kt(times, power, measured_ghi=None, fit_mask=np.ones(len(times), bool))


def test_legacy_method_still_reproducible(site):
    times, power, _, _ = site
    kt, p_cs = _calc(power_model="legacy_correlation").calculate_kt(times, power)
    day = (p_cs >= 1.0).to_numpy()
    # documents the saturation of the old envelope (daytime share at the clip)
    assert np.isclose(kt[day], 1.5).mean() > 0.05
