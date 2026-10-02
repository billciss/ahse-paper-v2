"""Tests of the honest per-horizon-block selector."""
import numpy as np
import pytest

from src.models.honest_selection import HonestBlockSelector, default_blocks

H = 24


def _data(n=24 * 150, seed=0, night_frac=0.0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.2, 1, (n, H))
    night = rng.uniform(size=(n, H)) < night_frac
    y[night] = 0.0
    preds = {
        "A": y + rng.normal(size=(n, H)) * np.where(np.arange(H) < 1, 0.02, 0.15),  # good at t+1 only
        "B": y + rng.normal(size=(n, H)) * 0.10,
        "C": y + rng.normal(size=(n, H)) * 0.11,
    }
    return y, preds, y > 0, np.arange(n) // 24


def test_blocks_cover_horizon():
    b = default_blocks(96, 4)
    assert b == {"first_hour": (0, 4), "1_to_6h": (4, 24), "6h_to_end": (24, 96)}


def test_picks_short_lead_specialist_and_average_elsewhere():
    y, p, m, d = _data()
    sel = HonestBlockSelector(default_blocks(H, 1)).fit(p, y, m, d)
    assert sel.decisions["first_hour"]["chosen"] == ["A"]
    assert len(sel.decisions["6h_to_end"]["chosen"]) >= 2
    assert all(dec["lcb_gain"] > 0 for dec in sel.decisions.values() if dec["chosen"] != [sel.reference])


def test_guard_keeps_reference_without_real_gain():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.2, 1, (24 * 60, H))
    noise = rng.normal(size=y.shape) * 0.1
    p = {"A": y + noise, "B": y + noise + rng.normal(size=y.shape) * 1e-4}   # practically identical
    sel = HonestBlockSelector(default_blocks(H, 1), top_k=(2,)).fit(p, y, y > 0, np.arange(len(y)) // 24)
    assert all(dec["chosen"] == [sel.reference] for dec in sel.decisions.values())


def test_predict_composes_blocks_and_requires_fit():
    y, p, m, d = _data()
    sel = HonestBlockSelector(default_blocks(H, 1))
    with pytest.raises(RuntimeError):
        sel.predict(p)
    sel.fit(p, y, m, d)
    out = sel.predict(p)
    a, b = sel.decisions["first_hour"]["steps"]
    np.testing.assert_allclose(out[:, a:b], np.mean([p[k] for k in sel.decisions["first_hour"]["chosen"]], axis=0)[:, a:b])


def test_night_samples_do_not_drive_selection():
    """A model perfect at night only must not win: errors are computed on daytime targets."""
    y, p, m, d = _data(night_frac=0.5)
    p["NightOnly"] = np.where(y == 0, 0.0, y + 0.5)          # perfect at night, bad by day
    sel = HonestBlockSelector(default_blocks(H, 1)).fit(p, y, m, d)
    assert sel.reference != "NightOnly"
    assert all("NightOnly" not in dec["chosen"] for dec in sel.decisions.values())
