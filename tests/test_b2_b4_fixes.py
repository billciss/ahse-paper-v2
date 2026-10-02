"""Regression tests for audit fixes B2 (persistence baselines, single FSS) and B4 (honest stacking)."""
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics_factory import MetricsFactory
from src.models.baselines import PersistenceModel, SmartPersistenceModel
from src.models.ensembles import AHSEMultiHorizon

KT_IDX = 2


def _scaled_windows(kt_windows, n_features=4, seed=0):
    """Build scaled (n, L, F) inputs whose raw feature KT_IDX equals kt_windows."""
    rng = np.random.default_rng(seed)
    n, L = kt_windows.shape
    raw = rng.normal(size=(n, L, n_features))
    raw[:, :, KT_IDX] = kt_windows
    scaler = StandardScaler().fit(raw.reshape(-1, n_features))
    scaled = scaler.transform(raw.reshape(-1, n_features)).reshape(n, L, n_features)
    return scaled, scaler


# ---------------------------------------------------------------- B2: baselines
def test_smart_persistence_repeats_same_slot_of_previous_day():
    L = H = P = 8
    kt = np.arange(3 * L, dtype=float).reshape(3, L) / 30
    X, sc = _scaled_windows(kt)
    pred = SmartPersistenceModel(pred_len=H, seasonal_period=P).predict(
        X, feature_scaler=sc, kt_feature_idx=KT_IDX)
    np.testing.assert_allclose(pred, kt, atol=1e-9)


def test_smart_persistence_differs_from_last_value_persistence():
    L = H = P = 8
    kt = np.tile(np.linspace(0, 1, L), (2, 1))
    X, sc = _scaled_windows(kt)
    smart = SmartPersistenceModel(pred_len=H, seasonal_period=P).predict(
        X, feature_scaler=sc, kt_feature_idx=KT_IDX)
    naive = PersistenceModel(pred_len=H).predict(X, feature_scaler=sc, kt_feature_idx=KT_IDX)
    assert not np.allclose(smart, naive)


def test_smart_persistence_wraps_when_horizon_exceeds_period():
    L, H, P = 6, 6, 3  # h = 0..5 -> window slots 3,4,5,3,4,5
    kt = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    X, sc = _scaled_windows(kt)
    pred = SmartPersistenceModel(pred_len=H, seasonal_period=P).predict(
        X, feature_scaler=sc, kt_feature_idx=KT_IDX)
    np.testing.assert_allclose(pred[0], [0.3, 0.4, 0.5, 0.3, 0.4, 0.5], atol=1e-9)


def test_smart_persistence_requires_kt_column_and_long_enough_window():
    X, sc = _scaled_windows(np.zeros((1, 4)))
    with pytest.raises(ValueError):
        SmartPersistenceModel(pred_len=4, seasonal_period=4).predict(X)
    with pytest.raises(ValueError):
        SmartPersistenceModel(pred_len=4, seasonal_period=8).predict(
            X, feature_scaler=sc, kt_feature_idx=KT_IDX)


# ---------------------------------------------------------------- B2: single FSS
def test_fss_matches_definition_on_masked_samples():
    rng = np.random.default_rng(1)
    y = rng.uniform(size=(50, 4))
    ref = y + rng.normal(scale=0.3, size=y.shape)
    pred = y + rng.normal(scale=0.1, size=y.shape)
    mask = rng.uniform(size=y.shape) > 0.3
    expected = 1 - np.mean((y - pred)[mask] ** 2) / np.mean((y - ref)[mask] ** 2)
    assert MetricsFactory.forecast_skill_score(y, pred, ref, mask) == pytest.approx(expected)


def test_fss_rejects_misaligned_reference():
    y = np.zeros((10, 4))
    with pytest.raises(ValueError):
        MetricsFactory.forecast_skill_score(y, y, np.zeros((10, 3)))
    with pytest.raises(ValueError):
        MetricsFactory.forecast_skill_score(y, y, y, mask=np.ones((10, 3), dtype=bool))


def test_compute_all_reports_both_skill_scores():
    rng = np.random.default_rng(2)
    y = rng.uniform(size=(40, 4))
    pred = y + 0.05
    pers = y + 0.2
    smart = y + 0.1
    mask = np.ones_like(y, dtype=bool)
    out = MetricsFactory.compute_all(y, pred, mask, y_persistence=pers, y_smart_persistence=smart)
    assert out["Skill_Score"] == pytest.approx(1 - 0.05 ** 2 / 0.2 ** 2)
    assert out["Skill_Score_Daily"] == pytest.approx(1 - 0.05 ** 2 / 0.1 ** 2)


# ---------------------------------------------------------------- B4: honest stacking
def _val_problem(n=400, H=3, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.uniform(size=(n, H))
    preds = {m: y + rng.normal(scale=s, size=y.shape) for m, s in [("A", 0.10), ("B", 0.11), ("C", 0.12)]}
    return y, preds


def test_stacking_selection_predictions_are_out_of_fold():
    """Changing the targets of one validation block must not change that block's stacking predictions."""
    y, preds = _val_problem()
    ahse = AHSEMultiHorizon(stacking_folds=4)
    oof = ahse._fit_stacking(preds, y, ["A", "B", "C"])

    y2 = y.copy()
    y2[:100] += 5.0  # first block only
    oof2 = AHSEMultiHorizon(stacking_folds=4)._fit_stacking(preds, y2, ["A", "B", "C"])
    np.testing.assert_allclose(oof[:100], oof2[:100])
    assert not np.allclose(oof[100:], oof2[100:])


def test_stacking_refits_on_full_validation_for_test():
    y, preds = _val_problem()
    ahse = AHSEMultiHorizon(stacking_folds=4)
    ahse._fit_stacking(preds, y, ["A", "B", "C"])
    assert len(ahse.meta_learners) == y.shape[1]
    X0 = np.column_stack([preds[m][:, 0] for m in "ABC"])
    from sklearn.linear_model import Ridge
    full = Ridge(alpha=1.0).fit(X0, y[:, 0])
    np.testing.assert_allclose(ahse.meta_learners[0].coef_, full.coef_)


def test_r2_is_not_clipped():
    y = np.linspace(0, 1, 40).reshape(10, 4)
    out = MetricsFactory.compute_all(y, 1 - y + 3.0, np.ones_like(y, dtype=bool))
    assert out["R2"] < -1.0
