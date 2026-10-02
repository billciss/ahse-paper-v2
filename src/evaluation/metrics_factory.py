"""
Metrics Factory for PV Forecasting Evaluation
==============================================
Provides rigorous scientific evaluation with daytime masking.

Key Features:
- Daytime masking (GHI > threshold) for honest evaluation
- Skill Score relative to Persistence baseline
- Multi-horizon error analysis
- Physical unit conversion (kt -> Watts)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    rmse: float
    mae: float
    r2: float
    skill_score: float
    mape: Optional[float] = None
    smape: Optional[float] = None
    n_samples: int = 0
    n_masked: int = 0
    model_name: str = ""
    horizon: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'Model': self.model_name,
            'Horizon': self.horizon,
            'RMSE': self.rmse,
            'MAE': self.mae,
            'R²': self.r2,
            'Skill Score': self.skill_score,
            'MAPE (%)': self.mape,
            'sMAPE (%)': self.smape,
            'N_samples': self.n_samples,
            'N_masked': self.n_masked
        }


class DaylightMaskCalculator:
    """
    Calculates daytime mask for honest evaluation.
    
    Scientific justification:
    - Night predictions are trivially 0
    - Including night inflates R² artificially
    - Daytime-only metrics reflect true forecasting skill
    """
    
    def __init__(self, ghi_threshold: float = 10.0):
        """
        Args:
            ghi_threshold: Minimum GHI (W/m²) to consider as daytime
        """
        self.ghi_threshold = ghi_threshold
    
    def calculate_mask(
        self,
        ghi: np.ndarray,
        clear_sky_power: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate daytime mask.
        
        Args:
            ghi: Global Horizontal Irradiance values
            clear_sky_power: Optional clear sky power (used if GHI unavailable)
        
        Returns:
            Boolean mask where True = daytime
        """
        if ghi is not None:
            mask = np.array(ghi) > self.ghi_threshold
        elif clear_sky_power is not None:
            # Use clear sky power as proxy (power > 0 indicates day)
            mask = np.array(clear_sky_power) > 0
        else:
            # No mask available, use all data
            logger.warning("No GHI or clear_sky_power available, using all data")
            mask = np.ones(len(ghi), dtype=bool)
        
        return mask
    
    def calculate_multi_horizon_mask(
        self,
        ghi: np.ndarray,
        horizon_length: int
    ) -> np.ndarray:
        """
        Calculate mask for multi-horizon predictions.
        
        For each sample, check if ANY hour in the horizon has daylight.
        
        Args:
            ghi: GHI values for all timesteps
            horizon_length: Number of steps in prediction horizon
        
        Returns:
            Mask of shape (n_samples, horizon_length)
        """
        n_samples = len(ghi) - horizon_length + 1
        mask = np.zeros((n_samples, horizon_length), dtype=bool)
        
        for i in range(n_samples):
            mask[i] = ghi[i:i+horizon_length] > self.ghi_threshold
        
        return mask


class MetricsCalculator:
    """
    Calculates evaluation metrics with optional daytime masking.
    """
    
    def __init__(
        self,
        use_daytime_mask: bool = True,
        ghi_threshold: float = 10.0,
        epsilon: float = 1e-8
    ):
        """
        Args:
            use_daytime_mask: Whether to apply daytime masking
            ghi_threshold: GHI threshold for daytime (W/m²)
            epsilon: Small value to prevent division by zero
        """
        self.use_daytime_mask = use_daytime_mask
        self.mask_calculator = DaylightMaskCalculator(ghi_threshold)
        self.epsilon = epsilon
    
    def calculate_rmse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """Calculate Root Mean Square Error."""
        y_true, y_pred = self._apply_mask(y_true, y_pred, mask)
        if len(y_true) == 0:
            return np.nan
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mae(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """Calculate Mean Absolute Error."""
        y_true, y_pred = self._apply_mask(y_true, y_pred, mask)
        if len(y_true) == 0:
            return np.nan
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_r2(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate R² (Coefficient of Determination).
        
        R² = 1 - SS_res / SS_tot
        """
        y_true, y_pred = self._apply_mask(y_true, y_pred, mask)
        if len(y_true) == 0:
            return np.nan
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot < self.epsilon:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return max(-1.0, min(1.0, r2))  # Clip to [-1, 1]
    
    def calculate_mape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None,
        threshold: float = 10.0
    ) -> Optional[float]:
        """
        Calculate Mean Absolute Percentage Error.
        Only calculated where y_true > threshold.
        """
        y_true, y_pred = self._apply_mask(y_true, y_pred, mask)
        
        valid = y_true > threshold
        if valid.sum() < 10:
            return None
        
        mape = np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100
        return float(mape) if mape < 1000 else None
    
    def calculate_smape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        More robust than MAPE for small values.
        """
        y_true, y_pred = self._apply_mask(y_true, y_pred, mask)
        if len(y_true) == 0:
            return np.nan
        
        denominator = np.abs(y_true) + np.abs(y_pred) + self.epsilon
        smape = 200 * np.mean(np.abs(y_true - y_pred) / denominator)
        return float(smape)
    
    def calculate_skill_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_persistence: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Skill Score relative to Persistence baseline.
        
        SS = 1 - (MSE_model / MSE_persistence)
        
        Interpretation:
        - SS > 0: Model beats persistence
        - SS = 0: Model equals persistence  
        - SS < 0: Model worse than persistence
        - SS = 1: Perfect forecast
        
        Args:
            y_true: Ground truth values
            y_pred: Model predictions
            y_persistence: Persistence baseline predictions
            mask: Optional daytime mask
        """
        y_true_m, y_pred_m = self._apply_mask(y_true, y_pred, mask)
        _, y_pers_m = self._apply_mask(y_true, y_persistence, mask)
        
        if len(y_true_m) == 0:
            return np.nan
        
        mse_model = np.mean((y_true_m - y_pred_m) ** 2)
        mse_persistence = np.mean((y_true_m - y_pers_m) ** 2)
        
        if mse_persistence < self.epsilon:
            return 0.0
        
        skill_score = 1 - (mse_model / mse_persistence)
        return float(skill_score)
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_persistence: Optional[np.ndarray] = None,
        ghi: Optional[np.ndarray] = None,
        model_name: str = "",
        horizon: str = ""
    ) -> MetricsResult:
        """
        Calculate all metrics with optional daytime masking.
        
        Args:
            y_true: Ground truth values
            y_pred: Model predictions
            y_persistence: Persistence baseline for Skill Score
            ghi: GHI values for daytime masking
            model_name: Name of the model
            horizon: Forecast horizon description
        
        Returns:
            MetricsResult with all calculated metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Calculate daytime mask
        if self.use_daytime_mask and ghi is not None:
            mask = self.mask_calculator.calculate_mask(ghi)
        else:
            mask = np.ones(len(y_true), dtype=bool)
        
        # Ensure arrays have same length
        min_len = min(len(y_true), len(y_pred), len(mask))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        mask = mask[:min_len]
        
        n_masked = (~mask).sum()
        n_samples = mask.sum()
        
        # Calculate metrics
        rmse = self.calculate_rmse(y_true, y_pred, mask)
        mae = self.calculate_mae(y_true, y_pred, mask)
        r2 = self.calculate_r2(y_true, y_pred, mask)
        mape = self.calculate_mape(y_true, y_pred, mask)
        smape = self.calculate_smape(y_true, y_pred, mask)
        
        # Skill Score
        if y_persistence is not None:
            y_persistence = np.array(y_persistence).flatten()[:min_len]
            skill_score = self.calculate_skill_score(y_true, y_pred, y_persistence, mask)
        else:
            skill_score = np.nan
        
        return MetricsResult(
            rmse=rmse,
            mae=mae,
            r2=r2,
            skill_score=skill_score,
            mape=mape,
            smape=smape,
            n_samples=n_samples,
            n_masked=n_masked,
            model_name=model_name,
            horizon=horizon
        )
    
    def _apply_mask(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply mask to arrays."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        if mask is not None:
            mask = np.array(mask).flatten()
            min_len = min(len(y_true), len(y_pred), len(mask))
            y_true = y_true[:min_len][mask[:min_len]]
            y_pred = y_pred[:min_len][mask[:min_len]]
        
        return y_true, y_pred


class MultiHorizonMetrics:
    """
    Calculates metrics for multi-horizon predictions.
    
    Analyzes how prediction error evolves across the forecast horizon.
    """
    
    def __init__(
        self,
        horizon_steps: int = 96,
        resolution_minutes: int = 15,
        use_daytime_mask: bool = True,
        ghi_threshold: float = 10.0
    ):
        """
        Args:
            horizon_steps: Number of prediction steps (H)
            resolution_minutes: Time resolution in minutes
            use_daytime_mask: Whether to apply daytime masking
            ghi_threshold: GHI threshold for daytime
        """
        self.horizon_steps = horizon_steps
        self.resolution_minutes = resolution_minutes
        self.calculator = MetricsCalculator(use_daytime_mask, ghi_threshold)
    
    def calculate_per_step_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ghi: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Calculate metrics for each step in the horizon.
        
        Args:
            y_true: Shape (n_samples, horizon_steps)
            y_pred: Shape (n_samples, horizon_steps)
            ghi: Optional GHI values for masking
        
        Returns:
            DataFrame with metrics per horizon step
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        n_samples, n_steps = y_true.shape
        
        results = []
        for step in range(n_steps):
            step_true = y_true[:, step]
            step_pred = y_pred[:, step]
            
            # Calculate horizon in hours
            horizon_hours = (step + 1) * self.resolution_minutes / 60
            
            # Get GHI for this step if available
            step_ghi = ghi[:, step] if ghi is not None and ghi.ndim > 1 else None
            
            result = self.calculator.calculate_all_metrics(
                step_true, step_pred,
                ghi=step_ghi,
                horizon=f"+{horizon_hours:.1f}h"
            )
            
            results.append({
                'Step': step + 1,
                'Horizon_minutes': (step + 1) * self.resolution_minutes,
                'Horizon_hours': horizon_hours,
                'RMSE': result.rmse,
                'MAE': result.mae,
                'R²': result.r2,
                'N_samples': result.n_samples
            })
        
        return pd.DataFrame(results)
    
    def calculate_aggregated_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_persistence: Optional[np.ndarray] = None,
        ghi: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate aggregated metrics across all horizons.
        
        Aggregation methods:
        - Mean: Average error across all steps
        - Step-weighted: Weight by horizon distance
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        if ghi is not None:
            ghi = np.array(ghi).flatten()
        
        result = self.calculator.calculate_all_metrics(
            y_true, y_pred,
            y_persistence=y_persistence.flatten() if y_persistence is not None else None,
            ghi=ghi
        )
        
        return result.to_dict()
    
    def analyze_error_growth(
        self,
        per_step_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Analyze how errors grow with forecast horizon.
        
        Returns:
            - error_growth_rate: Linear regression slope
        - r2_degradation: Rate of R² decrease
        - critical_horizon: Hours where R² drops below 0.5
        """
        steps = per_step_df['Horizon_hours'].values
        rmse = per_step_df['RMSE'].values
        r2 = per_step_df['R²'].values
        
        # Linear fit for RMSE growth
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(steps, rmse)
        
        # Find critical horizon (R² < 0.5)
        critical_mask = r2 < 0.5
        if critical_mask.any():
            critical_horizon = steps[critical_mask][0]
        else:
            critical_horizon = steps[-1]
        
        return {
            'rmse_growth_rate_per_hour': slope,
            'rmse_initial': intercept,
            'error_growth_r2': r_value ** 2,
            'critical_horizon_hours': critical_horizon,
            'final_r2': r2[-1] if len(r2) > 0 else np.nan
        }


class ModelComparison:
    """
    Compares multiple models and generates ranking tables.
    """
    
    def __init__(self, use_daytime_mask: bool = True, ghi_threshold: float = 10.0):
        self.calculator = MetricsCalculator(use_daytime_mask, ghi_threshold)
    
    def compare_models(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        y_persistence: Optional[np.ndarray] = None,
        ghi: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compare all models and return ranked DataFrame.
        
        Args:
            predictions: Dict of {model_name: predictions}
            y_true: Ground truth
            y_persistence: Persistence baseline
            ghi: GHI for daytime masking
        
        Returns:
            DataFrame with models ranked by RMSE
        """
        results = []
        
        for model_name, y_pred in predictions.items():
            result = self.calculator.calculate_all_metrics(
                y_true, y_pred,
                y_persistence=y_persistence,
                ghi=ghi,
                model_name=model_name
            )
            results.append(result.to_dict())
        
        df = pd.DataFrame(results)
        df = df.sort_values('RMSE')
        df['Rank'] = range(1, len(df) + 1)
        
        return df[['Rank', 'Model', 'RMSE', 'MAE', 'R²', 'Skill Score', 'sMAPE (%)', 'N_samples']]
    
    def statistical_significance_test(
        self,
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform Diebold-Mariano test for comparing forecast accuracy.
        
        Args:
            errors_model1: Prediction errors from model 1
            errors_model2: Prediction errors from model 2
            alpha: Significance level
        
        Returns:
            Test results with p-value and conclusion
        """
        from scipy import stats
        
        # Calculate squared error differences
        d = errors_model1 ** 2 - errors_model2 ** 2
        
        # DM statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        n = len(d)
        
        if var_d < 1e-10:
            return {
                'dm_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'conclusion': 'Models are equivalent'
            }
        
        dm_stat = mean_d / np.sqrt(var_d / n)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        significant = p_value < alpha
        
        if significant:
            if mean_d < 0:
                conclusion = "Model 1 is significantly better"
            else:
                conclusion = "Model 2 is significantly better"
        else:
            conclusion = "No significant difference"
        
        return {
            'dm_statistic': float(dm_stat),
            'p_value': float(p_value),
            'significant': significant,
            'conclusion': conclusion
        }


def load_metrics_config(config_path: str = "configs/data_config_yulara_neighbours.yaml") -> Dict:
    """Load metrics configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return {
        'use_daytime_mask': True,
        'ghi_threshold': config.get('preprocessing', {}).get('night_threshold_ghi', 10.0),
        'epsilon': 1e-8
    }


def create_metrics_calculator(config_path: str = "configs/data_config_yulara_neighbours.yaml") -> MetricsCalculator:
    """Factory function to create MetricsCalculator from config."""
    config = load_metrics_config(config_path)
    return MetricsCalculator(**config)


# Convenience function for simple evaluation
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ghi: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Simple function to evaluate predictions.
    
    Usage:
        metrics = evaluate_predictions(y_true, y_pred, ghi=ghi_test)
        print(f"R²: {metrics['R²']:.4f}")
    """
    calculator = MetricsCalculator(use_daytime_mask=ghi is not None)
    result = calculator.calculate_all_metrics(
        y_true, y_pred, ghi=ghi, model_name=model_name
    )
    return result.to_dict()


# =============================================================================
# MetricsFactory - Static interface for convenience
# =============================================================================

class MetricsFactory:
    """
    Static factory class for easy metric computation.
    
    Usage:
        metrics = MetricsFactory.compute_all(y_true, y_pred, daytime_mask=mask)
    """
    
    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        daytime_mask: Optional[np.ndarray] = None,
        ghi: Optional[np.ndarray] = None,
        model_name: str = "Model",
        y_persistence: Optional[np.ndarray] = None,
        y_smart_persistence: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics with optional daytime masking.

        Args:
            y_true        : Ground truth values
            y_pred        : Model predictions
            daytime_mask  : Boolean mask (True = include)
            ghi           : GHI values for automatic mask calculation
            model_name    : Name of model for reporting
            y_persistence : Predictions of the Persistence baseline model
                            (n_samples, pred_len) or (n_samples*pred_len,).
                            When provided, used as denominator for FSS.
                            When None, FSS is set to NaN (rather than using
                            the incorrect np.roll hack).
            y_smart_persistence : Predictions of the day-ahead (seasonal) persistence;
                            when provided, reported as Skill_Score_Daily.
        
        Returns:
            Dictionary with RMSE, MAE, R², Skill_Score, nRMSE
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Create mask
        if daytime_mask is not None:
            mask = np.array(daytime_mask).flatten()[:len(y_true)]
        elif ghi is not None:
            mask = np.array(ghi).flatten()[:len(y_true)] > 10.0
        else:
            mask = np.ones(len(y_true), dtype=bool)
        
        # Apply mask
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        
        if len(y_true_masked) == 0:
            return {
                'Model': model_name,
                'RMSE': np.nan,
                'MAE': np.nan,
                'R²': np.nan,
                'Skill_Score': np.nan,
                'Skill_Score_Daily': np.nan,
                'nRMSE': np.nan
            }
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_true_masked - y_pred_masked) ** 2))
        mae = np.mean(np.abs(y_true_masked - y_pred_masked))
        
        ss_res = np.sum((y_true_masked - y_pred_masked) ** 2)
        ss_tot = np.sum((y_true_masked - np.mean(y_true_masked)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        # R2 is reported unclipped: clipping at -1 hid how poor weak baselines are.
        
        # Forecast Skill Score — single definition, see forecast_skill_score().
        # References are flattened like y_true and must align one-to-one with it.
        def _fss(y_ref):
            if y_ref is None:
                return float('nan')
            return MetricsFactory.forecast_skill_score(
                y_true, y_pred, np.array(y_ref).flatten(), mask)

        skill_score = _fss(y_persistence)
        skill_score_daily = _fss(y_smart_persistence)
        
        # Normalized RMSE
        nrmse = rmse / (np.mean(y_true_masked) + 1e-8) * 100
        
        return {
            'Model': model_name,
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'Skill_Score': float(skill_score),
            'Skill_Score_Daily': float(skill_score_daily),
            'nRMSE': float(nrmse)
        }
    
    @staticmethod
    def forecast_skill_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_ref: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Forecast skill score FSS = 1 - MSE(model) / MSE(reference), computed on the
        same (optionally masked) samples for both terms.

        Shapes of y_true, y_pred, y_ref (and mask) must match exactly: a mismatch
        raises instead of being silently truncated, which is how misaligned
        persistence forecasts previously produced inconsistent FSS values.
        """
        y_true, y_pred, y_ref = (np.asarray(a, dtype=float) for a in (y_true, y_pred, y_ref))
        if not (y_true.shape == y_pred.shape == y_ref.shape):
            raise ValueError(
                f"FSS shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, y_ref {y_ref.shape}")
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != y_true.shape:
                raise ValueError(f"FSS mask shape {mask.shape} != y_true shape {y_true.shape}")
            y_true, y_pred, y_ref = y_true[mask], y_pred[mask], y_ref[mask]
        mse_ref = np.mean((y_true - y_ref) ** 2)
        if mse_ref <= 0:
            return float('nan')
        return float(1 - np.mean((y_true - y_pred) ** 2) / mse_ref)

    @staticmethod
    def compute_skill_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_baseline: np.ndarray
    ) -> float:
        """Compute Skill Score relative to baseline."""
        mse_model = np.mean((y_true - y_pred) ** 2)
        mse_baseline = np.mean((y_true - y_baseline) ** 2)
        return 1 - (mse_model / (mse_baseline + 1e-8))


# Alias for backward compatibility
compute_skill_score = MetricsFactory.compute_skill_score
