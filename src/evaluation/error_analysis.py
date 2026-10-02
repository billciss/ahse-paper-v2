"""
Error Analysis Module
=====================
Comprehensive analysis of prediction errors for scientific reporting.

Features:
- Error distribution analysis
- Horizon degradation analysis
- Weather regime performance breakdown
- Seasonal/temporal patterns
- Residual diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class ErrorAnalysisResult:
    """Container for error analysis results."""
    mean_error: float  # Bias
    std_error: float
    median_error: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    iqr: float
    outlier_ratio: float
    
    def to_dict(self) -> Dict:
        return {
            'Mean Error (Bias)': self.mean_error,
            'Std Error': self.std_error,
            'Median Error': self.median_error,
            'Q25': self.q25,
            'Q75': self.q75,
            'IQR': self.iqr,
            'Skewness': self.skewness,
            'Kurtosis': self.kurtosis,
            'Outlier Ratio (%)': self.outlier_ratio * 100
        }


class ErrorDistributionAnalyzer:
    """
    Analyzes the distribution of prediction errors.
    
    Important for:
    - Detecting systematic biases
    - Understanding error patterns
    - Checking model assumptions
    """
    
    def __init__(self, outlier_threshold: float = 3.0):
        """
        Args:
            outlier_threshold: Number of IQR to define outliers
        """
        self.outlier_threshold = outlier_threshold
    
    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> ErrorAnalysisResult:
        """
        Analyze error distribution.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            mask: Optional boolean mask
        
        Returns:
            ErrorAnalysisResult with distribution statistics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        if mask is not None:
            mask = np.array(mask).flatten()
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        
        errors = y_true - y_pred
        
        q25, q75 = np.percentile(errors, [25, 75])
        iqr = q75 - q25
        
        # Outliers using IQR method
        lower_bound = q25 - self.outlier_threshold * iqr
        upper_bound = q75 + self.outlier_threshold * iqr
        outliers = (errors < lower_bound) | (errors > upper_bound)
        
        return ErrorAnalysisResult(
            mean_error=float(np.mean(errors)),
            std_error=float(np.std(errors)),
            median_error=float(np.median(errors)),
            q25=float(q25),
            q75=float(q75),
            iqr=float(iqr),
            skewness=float(stats.skew(errors)),
            kurtosis=float(stats.kurtosis(errors)),
            outlier_ratio=float(outliers.mean())
        )
    
    def test_normality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        max_samples: int = 5000
    ) -> Dict:
        """
        Test if errors follow normal distribution.
        
        Uses Shapiro-Wilk test (limited to 5000 samples).
        """
        errors = np.array(y_true).flatten() - np.array(y_pred).flatten()
        
        # Subsample for Shapiro-Wilk if needed
        if len(errors) > max_samples:
            errors = np.random.choice(errors, max_samples, replace=False)
        
        try:
            stat, p_value = stats.shapiro(errors)
            normal = p_value > 0.05
        except Exception:
            stat, p_value = np.nan, np.nan
            normal = None
        
        return {
            'shapiro_statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': normal,
            'interpretation': 'Normal' if normal else 'Non-normal'
        }


class HorizonDegradationAnalyzer:
    """
    Analyzes how prediction quality degrades with forecast horizon.
    
    Key insights:
    - Error growth rate
    - Critical horizon (where skill drops significantly)
    - Optimal forecasting window
    """
    
    def __init__(
        self,
        resolution_minutes: int = 15,
        skill_threshold: float = 0.5
    ):
        """
        Args:
            resolution_minutes: Time resolution
            skill_threshold: R² below which forecast is considered unskillful
        """
        self.resolution_minutes = resolution_minutes
        self.skill_threshold = skill_threshold
    
    def analyze_degradation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Analyze error degradation across horizon steps.
        
        Args:
            y_true: Shape (n_samples, horizon_steps)
            y_pred: Shape (n_samples, horizon_steps)
            mask: Optional mask (n_samples, horizon_steps)
        
        Returns:
            DataFrame with per-step metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.ndim == 1:
            logger.warning("1D arrays provided, cannot analyze horizon degradation")
            return pd.DataFrame()
        
        n_samples, n_steps = y_true.shape
        
        results = []
        for step in range(n_steps):
            true_step = y_true[:, step]
            pred_step = y_pred[:, step]
            
            if mask is not None:
                step_mask = mask[:, step] if mask.ndim > 1 else mask
                true_step = true_step[step_mask]
                pred_step = pred_step[step_mask]
            
            horizon_hours = (step + 1) * self.resolution_minutes / 60
            
            # Metrics
            rmse = np.sqrt(np.mean((true_step - pred_step) ** 2))
            mae = np.mean(np.abs(true_step - pred_step))
            
            ss_res = np.sum((true_step - pred_step) ** 2)
            ss_tot = np.sum((true_step - np.mean(true_step)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            # Bias
            bias = np.mean(pred_step - true_step)
            
            results.append({
                'Step': step + 1,
                'Horizon_hours': horizon_hours,
                'RMSE': rmse,
                'MAE': mae,
                'R²': max(-1, min(1, r2)),
                'Bias': bias,
                'N_samples': len(true_step)
            })
        
        return pd.DataFrame(results)
    
    def find_critical_horizon(
        self,
        degradation_df: pd.DataFrame
    ) -> Dict:
        """
        Find the horizon where prediction skill degrades significantly.
        
        Returns:
            Critical horizons for different thresholds
        """
        hours = degradation_df['Horizon_hours'].values
        r2 = degradation_df['R²'].values
        
        results = {}
        
        # Find where R² drops below threshold
        for threshold in [0.9, 0.7, 0.5, 0.3]:
            below = r2 < threshold
            if below.any():
                critical_hour = hours[below][0]
            else:
                critical_hour = hours[-1]
            results[f'R²<{threshold}'] = float(critical_hour)
        
        # Fit exponential decay
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = curve_fit(
                exp_decay, hours, r2,
                p0=[1.0, 0.1, 0.0],
                bounds=([0, 0, -1], [2, 10, 1]),
                maxfev=1000
            )
            
            results['decay_rate'] = float(popt[1])
            results['asymptotic_r2'] = float(popt[2])
            
            # Half-life of R² decay
            if popt[1] > 0:
                results['r2_halflife_hours'] = float(np.log(2) / popt[1])
        except Exception:
            results['decay_rate'] = np.nan
            results['asymptotic_r2'] = np.nan
            results['r2_halflife_hours'] = np.nan
        
        return results
    
    def calculate_error_growth_rate(
        self,
        degradation_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate the rate at which error grows with horizon.
        
        Uses linear regression on RMSE vs horizon.
        """
        hours = degradation_df['Horizon_hours'].values
        rmse = degradation_df['RMSE'].values
        
        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(hours, rmse)
        
        # Also fit log-log for power law
        try:
            log_hours = np.log(hours + 0.01)
            log_rmse = np.log(rmse + 0.01)
            power_slope, power_intercept, _, _, _ = stats.linregress(log_hours, log_rmse)
        except Exception:
            power_slope = np.nan
            power_intercept = np.nan
        
        return {
            'linear_slope': float(slope),  # RMSE increase per hour
            'linear_intercept': float(intercept),  # Initial RMSE
            'linear_r2': float(r_value ** 2),
            'p_value': float(p_value),
            'power_law_exponent': float(power_slope),
            'error_doubles_in_hours': float(intercept / slope) if slope > 0 else np.inf
        }
    
    def compute_horizon_rmse(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute RMSE for each horizon step.
        
        Args:
            y_true: Shape (n_samples, horizon) or (n_samples,)
            y_pred: Shape (n_samples, horizon) or (n_samples,)
            mask: Optional boolean mask
        
        Returns:
            Array of RMSE values, one per horizon step
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.ndim == 1:
            # Single-step case
            if mask is not None:
                y_true = y_true[mask]
                y_pred = y_pred[mask]
            return np.array([np.sqrt(np.mean((y_true - y_pred) ** 2))])
        
        n_steps = y_true.shape[1]
        rmse_per_step = np.zeros(n_steps)
        
        for step in range(n_steps):
            true_step = y_true[:, step]
            pred_step = y_pred[:, step]
            
            if mask is not None:
                if mask.ndim > 1:
                    step_mask = mask[:, step]
                else:
                    step_mask = mask
                true_step = true_step[step_mask]
                pred_step = pred_step[step_mask]
            
            rmse_per_step[step] = np.sqrt(np.mean((true_step - pred_step) ** 2))
        
        return rmse_per_step


class WeatherRegimeErrorAnalysis:
    """
    Analyzes prediction errors by weather regime.
    
    Different weather conditions have different predictability:
    - Clear sky: Highly predictable
    - Partly cloudy: Moderate difficulty
    - Overcast: Predictable (low variability)
    - Variable: Most challenging
    """
    
    # GHI column name candidates
    GHI_COLS = ['GHI', 'ghi', 'Global_Horizontal_Radiation', 'Radiation']
    
    def __init__(self, ghi_threshold: float = 10.0):
        self.ghi_threshold = ghi_threshold
    
    def classify_regimes(
        self,
        ghi: np.ndarray,
        kt: Optional[np.ndarray] = None,
        window: int = 12
    ) -> np.ndarray:
        """
        Classify each timestep into a weather regime.
        
        Args:
            ghi: Global Horizontal Irradiance
            kt: Optional Clear Sky Index
            window: Rolling window for variability
        
        Returns:
            Array of regime labels
        """
        ghi = np.array(ghi).flatten()
        n = len(ghi)
        regimes = np.array(['unknown'] * n, dtype=object)
        
        # Night detection
        night_mask = ghi < self.ghi_threshold
        regimes[night_mask] = 'night'
        
        # Normalize GHI for daytime
        day_ghi = ghi[~night_mask]
        if len(day_ghi) == 0:
            return regimes
        
        ghi_p95 = np.percentile(day_ghi, 95)
        ghi_norm = np.clip(ghi / (ghi_p95 + 1e-8), 0, 1.2)
        
        # Variability (coefficient of variation in rolling window)
        ghi_series = pd.Series(ghi)
        rolling_std = ghi_series.rolling(window, center=True, min_periods=1).std()
        rolling_mean = ghi_series.rolling(window, center=True, min_periods=1).mean()
        variability = (rolling_std / (rolling_mean + 1e-8)).values
        
        # Classification logic
        for i in range(n):
            if regimes[i] == 'night':
                continue
            
            if variability[i] > 0.4:
                regimes[i] = 'variable'
            elif kt is not None:
                # Use kt for more accurate classification
                if kt[i] > 0.7:
                    regimes[i] = 'clear_sky'
                elif kt[i] < 0.3:
                    regimes[i] = 'cloudy'
                else:
                    regimes[i] = 'partly_cloudy'
            else:
                # Use normalized GHI
                if ghi_norm[i] > 0.75:
                    regimes[i] = 'clear_sky'
                elif ghi_norm[i] < 0.25:
                    regimes[i] = 'cloudy'
                else:
                    regimes[i] = 'partly_cloudy'
        
        return regimes
    
    def analyze_by_regime(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        regimes: np.ndarray,
        min_samples: int = 50
    ) -> pd.DataFrame:
        """
        Calculate metrics for each weather regime.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            regimes: Regime labels for each sample
            min_samples: Minimum samples to include regime
        
        Returns:
            DataFrame with per-regime metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        regimes = np.array(regimes).flatten()
        
        results = []
        unique_regimes = ['clear_sky', 'partly_cloudy', 'cloudy', 'variable']
        
        for regime in unique_regimes:
            mask = regimes == regime
            n = mask.sum()
            
            if n < min_samples:
                continue
            
            true_r = y_true[mask]
            pred_r = y_pred[mask]
            
            rmse = np.sqrt(np.mean((true_r - pred_r) ** 2))
            mae = np.mean(np.abs(true_r - pred_r))
            
            ss_res = np.sum((true_r - pred_r) ** 2)
            ss_tot = np.sum((true_r - np.mean(true_r)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8) if ss_tot > 0 else 0
            
            bias = np.mean(pred_r - true_r)
            
            results.append({
                'Regime': regime,
                'N_samples': int(n),
                'Proportion (%)': float(n / len(y_true) * 100),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R²': float(max(-1, min(1, r2))),
                'Bias': float(bias)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('RMSE')
        return df


class TemporalErrorAnalysis:
    """
    Analyzes error patterns across time.
    
    Detects:
    - Hourly patterns
    - Seasonal patterns
    - Trends
    """
    
    def analyze_by_hour(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Analyze errors by hour of day.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            timestamps: Datetime index
        
        Returns:
            DataFrame with hourly metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        errors = y_true - y_pred
        
        df = pd.DataFrame({
            'error': errors,
            'hour': timestamps.hour[:len(errors)]
        })
        
        hourly = df.groupby('hour').agg({
            'error': ['mean', 'std', 'count']
        }).round(4)
        
        hourly.columns = ['Mean_Error', 'Std_Error', 'N_samples']
        hourly = hourly.reset_index()
        
        return hourly
    
    def analyze_by_month(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Analyze errors by month.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            timestamps: Datetime index
        
        Returns:
            DataFrame with monthly metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'month': timestamps.month[:len(y_true)]
        })
        
        def calc_metrics(group):
            rmse = np.sqrt(np.mean((group['y_true'] - group['y_pred']) ** 2))
            mae = np.mean(np.abs(group['y_true'] - group['y_pred']))
            ss_res = np.sum((group['y_true'] - group['y_pred']) ** 2)
            ss_tot = np.sum((group['y_true'] - group['y_true'].mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            return pd.Series({
                'RMSE': rmse,
                'MAE': mae,
                'R²': max(-1, min(1, r2)),
                'N_samples': len(group)
            })
        
        monthly = df.groupby('month').apply(calc_metrics)
        monthly = monthly.reset_index()
        
        return monthly


class ResidualDiagnostics:
    """
    Diagnostic tests for model residuals.
    
    Checks for:
    - Autocorrelation
    - Heteroscedasticity
    - Non-linearity
    """
    
    def calculate_acf(
        self,
        residuals: np.ndarray,
        max_lag: int = 48
    ) -> pd.DataFrame:
        """
        Calculate Autocorrelation Function of residuals.
        
        Significant ACF indicates model is missing temporal patterns.
        """
        residuals = np.array(residuals).flatten()
        n = len(residuals)
        mean = np.mean(residuals)
        var = np.var(residuals)
        
        acf_values = []
        for lag in range(max_lag + 1):
            if lag >= n:
                break
            autocov = np.sum((residuals[:n-lag] - mean) * (residuals[lag:] - mean)) / n
            acf = autocov / var if var > 0 else 0
            acf_values.append({
                'Lag': lag,
                'ACF': acf,
                'CI_upper': 1.96 / np.sqrt(n),
                'CI_lower': -1.96 / np.sqrt(n)
            })
        
        return pd.DataFrame(acf_values)
    
    def test_heteroscedasticity(
        self,
        y_true: np.ndarray,
        residuals: np.ndarray
    ) -> Dict:
        """
        Test for heteroscedasticity (error variance depends on y).
        
        Uses Breusch-Pagan test.
        """
        y_true = np.array(y_true).flatten()
        residuals = np.array(residuals).flatten()
        
        # Squared residuals regression on y_true
        from scipy import stats as scipy_stats
        
        sq_residuals = residuals ** 2
        slope, intercept, r_value, p_value, _ = scipy_stats.linregress(y_true, sq_residuals)
        
        # Chi-square statistic
        n = len(residuals)
        chi2 = n * r_value ** 2
        p_chi2 = 1 - scipy_stats.chi2.cdf(chi2, 1)
        
        return {
            'chi2_statistic': float(chi2),
            'p_value': float(p_chi2),
            'heteroscedastic': p_chi2 < 0.05,
            'interpretation': 'Heteroscedastic' if p_chi2 < 0.05 else 'Homoscedastic'
        }
    
    def durbin_watson(self, residuals: np.ndarray) -> Dict:
        """
        Durbin-Watson test for autocorrelation.
        
        DW ≈ 2: No autocorrelation
        DW < 2: Positive autocorrelation
        DW > 2: Negative autocorrelation
        """
        residuals = np.array(residuals).flatten()
        
        diff_sq = np.sum(np.diff(residuals) ** 2)
        sq_sum = np.sum(residuals ** 2)
        
        dw = diff_sq / sq_sum if sq_sum > 0 else 2.0
        
        if dw < 1.5:
            interpretation = 'Significant positive autocorrelation'
        elif dw > 2.5:
            interpretation = 'Significant negative autocorrelation'
        else:
            interpretation = 'No significant autocorrelation'
        
        return {
            'durbin_watson': float(dw),
            'interpretation': interpretation
        }


def generate_error_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    ghi: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict:
    """
    Generate comprehensive error analysis report.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        timestamps: Optional datetime index
        ghi: Optional GHI for regime analysis
        model_name: Name of the model
    
    Returns:
        Dictionary with all analysis results
    """
    report = {'model_name': model_name}
    
    # Error distribution
    dist_analyzer = ErrorDistributionAnalyzer()
    report['distribution'] = dist_analyzer.analyze(y_true, y_pred).to_dict()
    report['normality_test'] = dist_analyzer.test_normality(y_true, y_pred)
    
    # Residual diagnostics
    residuals = np.array(y_true).flatten() - np.array(y_pred).flatten()
    diagnostics = ResidualDiagnostics()
    report['durbin_watson'] = diagnostics.durbin_watson(residuals)
    report['heteroscedasticity'] = diagnostics.test_heteroscedasticity(y_true, residuals)
    
    # Temporal analysis
    if timestamps is not None:
        temporal = TemporalErrorAnalysis()
        report['hourly_errors'] = temporal.analyze_by_hour(y_true, y_pred, timestamps).to_dict()
        report['monthly_errors'] = temporal.analyze_by_month(y_true, y_pred, timestamps).to_dict()
    
    # Weather regime analysis
    if ghi is not None:
        regime_analyzer = WeatherRegimeErrorAnalysis()
        regimes = regime_analyzer.classify_regimes(ghi)
        report['regime_analysis'] = regime_analyzer.analyze_by_regime(y_true, y_pred, regimes).to_dict()
    
    return report
