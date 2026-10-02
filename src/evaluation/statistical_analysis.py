#!/usr/bin/env python3
"""
================================================================================
STATISTICAL ANALYSIS MODULE FOR SOLAR FORECASTING THESIS
================================================================================
Master's Thesis - Electrical Engineering, Wuhan University
Research: AHSE (Adaptive Honest Selection Ensemble) vs SOTA Models

This module provides:
- Diebold-Mariano Test for statistical significance
- Residual Analysis (Mean, Std, Skewness, Kurtosis, Kolmogorov-Smirnov)
- Feature Importance (XAI) using XGBoost/LightGBM gains
- Training Health Analysis (Loss Curves)

Author: MSc Candidate
Date: 2025
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, skew, kurtosis
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# HAC Newey-West via statsmodels (requis pour compute_dm_statistic corrigé)
try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    warnings.warn(
        "statsmodels non disponible — compute_dm_statistic utilisera "
        "la correction HAC manuelle (dégradée). Installez statsmodels>=0.13."
    )


class DieboldMarianoTest:
    """
    Diebold-Mariano Test for comparing forecast accuracy.
    
    Tests H0: Two forecasts have the same accuracy
    against H1: AHSE has significantly lower forecast error.
    
    Reference:
    Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy.
    Journal of Business & Economic Statistics.
    """
    
    @staticmethod
    def compute_dm_statistic(
        e1: np.ndarray,
        e2: np.ndarray,
        h: int = 1,
        loss_type: str = 'squared',
        hac_nlags: int = 96,
    ) -> Tuple[float, float]:
        """
        Compute Diebold-Mariano test statistic with HAC Newey-West correction.

        Correction du bug original : la boucle `for k in range(1, h)` avec h=1
        était vide → aucune correction HAC appliquée. Le DM stat était biaisé
        (surestimé) pour des séries autocorrélées.

        Cette implémentation utilise statsmodels OLS + cov_type='HAC' (Newey-West)
        avec nlags=96 (= un cycle diurne pour données 15-min). Le t-stat de la
        régression de d sur une constante est mathématiquement équivalent au DM
        stat corrigé HAC.

        Parameters
        ----------
        e1 : np.ndarray
            Erreurs du modèle benchmark (shape: n_samples × horizon ou 1D aplati)
        e2 : np.ndarray
            Erreurs du modèle proposé (même shape)
        h : int
            Horizon de prévision (conservé pour rétrocompatibilité ; hac_nlags
            est utilisé à la place pour la correction HAC)
        loss_type : str
            'squared' → différentiel MSE | 'absolute' → différentiel MAE
        hac_nlags : int
            Nombre de lags Newey-West. Défaut = 96 (cycle diurne 15-min).
            Pour données 1h : utiliser 24.

        Returns
        -------
        dm_stat : float
            Statistique DM (> 0 → modèle 2 meilleur que modèle 1)
        p_value : float
            P-valeur bilatérale
        """
        e1 = np.asarray(e1).flatten()
        e2 = np.asarray(e2).flatten()

        # Différentiel de perte
        if loss_type == 'squared':
            d = e1 ** 2 - e2 ** 2
        else:
            d = np.abs(e1) - np.abs(e2)

        n = len(d)
        d_bar = float(np.mean(d))

        if _HAS_STATSMODELS:
            # ── Implémentation statsmodels (recommandée) ─────────────────
            # Régression de d sur une constante → t-stat = DM stat HAC
            X = np.ones((n, 1))
            model = sm.OLS(d, X)
            res   = model.fit(cov_type='HAC',
                              cov_kwds={'maxlags': hac_nlags, 'use_correction': True},
                              use_t=False)
            dm_stat = float(res.tvalues[0])
            p_value = float(res.pvalues[0])
        else:
            # ── Fallback manuel (corrigé) ─────────────────────────────────
            # Newey-West à hac_nlags lags, formule directe
            gamma_0 = float(np.var(d, ddof=1))
            gamma_sum = 0.0
            for k in range(1, hac_nlags + 1):
                w_k = 1.0 - k / (hac_nlags + 1)   # pondération Bartlett
                if len(d) > k:
                    gamma_k = float(np.cov(d[:-k], d[k:])[0, 1])
                    gamma_sum += 2 * w_k * gamma_k
            var_d = max(gamma_0 + gamma_sum, 1e-12)
            dm_stat = d_bar / np.sqrt(var_d / n)
            p_value = float(2 * (1 - stats.norm.cdf(np.abs(dm_stat))))

        return dm_stat, p_value
    
    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Correction Benjamini-Hochberg (FDR) pour comparaisons multiples.

        Retourne un array de p-valeurs ajustées (BH-corrected).
        Une comparaison est significative si p_adj <= alpha.
        """
        n = len(p_values)
        order = np.argsort(p_values)
        ranks = np.argsort(order) + 1          # rangs 1-based
        p_adj = np.minimum(1.0, p_values * n / ranks)
        # Garantit la monotonie (BH standard)
        p_adj_mono = np.minimum.accumulate(p_adj[order[::-1]])[::-1]
        result = np.empty(n)
        result[order] = p_adj_mono
        return result

    @staticmethod
    def run_full_dm_matrix(
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        loss_type: str = 'squared',
        hac_nlags: int = 96,
        alpha: float = 0.05,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Matrice DM complète pour toutes les paires de modèles.

        Ferme les limitations #3 et #6 du plan d'article Q1 :
          - DM test avec correction HAC Newey-West (lag=96)
          - Correction Benjamini-Hochberg pour les comparaisons multiples
          - Matrice symétrique complète (pas seulement AHSE vs X)

        Parameters
        ----------
        predictions : Dict[str, np.ndarray]
        y_true      : np.ndarray
        loss_type   : 'squared' | 'absolute'
        hac_nlags   : int  — lags HAC (96 pour 15-min, 24 pour 1h)
        alpha       : float — seuil FDR (défaut 0.05)

        Returns
        -------
        dm_matrix   : pd.DataFrame — matrice des statistiques DM (i vs j)
        p_matrix    : pd.DataFrame — matrice des p-valeurs brutes
        p_adj_matrix: pd.DataFrame — matrice des p-valeurs BH-corrigées
        """
        models = list(predictions.keys())
        n_models = len(models)
        errors = {m: (y_true - predictions[m]).flatten() for m in models}

        dm_vals = np.zeros((n_models, n_models))
        p_vals  = np.ones((n_models, n_models))

        # Remplissage de la matrice triangulaire supérieure
        pairs_p = []   # pour BH
        pairs_idx = []

        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    continue
                dm, p = DieboldMarianoTest.compute_dm_statistic(
                    errors[models[i]], errors[models[j]],
                    loss_type=loss_type, hac_nlags=hac_nlags,
                )
                dm_vals[i, j] = dm
                p_vals[i, j]  = p
                if i < j:        # ne collecter qu'une fois par paire
                    pairs_p.append(p)
                    pairs_idx.append((i, j))

        # Correction BH sur les paires uniques
        if pairs_p:
            p_adj_flat = DieboldMarianoTest._benjamini_hochberg(
                np.array(pairs_p), alpha=alpha
            )
        else:
            p_adj_flat = np.array(pairs_p)

        p_adj_vals = np.ones((n_models, n_models))
        for (i, j), p_adj in zip(pairs_idx, p_adj_flat):
            p_adj_vals[i, j] = p_adj
            p_adj_vals[j, i] = p_adj   # symétrique

        dm_matrix    = pd.DataFrame(dm_vals,    index=models, columns=models)
        p_matrix     = pd.DataFrame(p_vals,     index=models, columns=models)
        p_adj_matrix = pd.DataFrame(p_adj_vals, index=models, columns=models)

        return dm_matrix, p_matrix, p_adj_matrix

    @staticmethod
    def plot_dm_heatmap(
        dm_matrix: pd.DataFrame,
        p_adj_matrix: pd.DataFrame,
        alpha: float = 0.05,
        save_path: Optional[str] = None,
        title: str = "Matrice Diebold-Mariano (HAC Newey-West, BH corrigé)",
    ):
        """
        Heatmap de la matrice DM avec masque de significativité BH.

        Cellules significatives (p_adj ≤ alpha) : valeur DM affichée.
        Cellules non-significatives : affichées en grisé.
        Convention : DM(i,j) > 0 → modèle j meilleur que modèle i.
        """
        models = dm_matrix.index.tolist()
        n = len(models)
        dm_vals    = dm_matrix.values.copy()
        sig_mask   = (p_adj_matrix.values <= alpha) & (~np.eye(n, dtype=bool))

        # Mettre la diagonale à NaN
        np.fill_diagonal(dm_vals, np.nan)

        fig, ax = plt.subplots(figsize=(max(7, n), max(6, n - 1)))

        # Fond grisé pour les valeurs non-significatives
        dm_display = np.where(sig_mask | np.eye(n, dtype=bool), dm_vals, np.nan)
        dm_nonsig  = np.where(~sig_mask & ~np.eye(n, dtype=bool), dm_vals, np.nan)

        im_bg  = ax.imshow(dm_nonsig,  cmap="Greys",   vmin=-5, vmax=5, alpha=0.3, aspect="auto")
        im_sig = ax.imshow(dm_display, cmap="RdBu_r",  vmin=-8, vmax=8, alpha=0.9, aspect="auto")

        plt.colorbar(im_sig, ax=ax, label="DM statistic", shrink=0.8)

        # Annotations
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#555")
                    continue
                val = dm_vals[i, j]
                if np.isnan(val):
                    continue
                is_sig = sig_mask[i, j]
                txt = f"{val:.2f}{'*' if is_sig else ''}"
                color = "white" if abs(val) > 4 and is_sig else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color,
                        fontweight="bold" if is_sig else "normal")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlabel("Modèle j (meilleur si DM > 0)", fontsize=9)
        ax.set_ylabel("Modèle i (benchmark)", fontsize=9)

        fig.text(0.01, 0.01,
                 f"* p_adj ≤ {alpha} (BH) | HAC lags={96} | Rouge=j meilleur, Bleu=i meilleur",
                 fontsize=7, color="#555")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            fig.savefig(str(save_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")

        return fig, ax

    @staticmethod
    def run_pairwise_tests(
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        reference_model: str = 'AHSE',
        h: int = 1,
        hac_nlags: int = 96,
    ) -> pd.DataFrame:
        """
        Run DM tests comparing reference model against all others.

        Parameters
        ----------
        predictions : Dict[str, np.ndarray]
            Dictionary of model predictions
        y_true : np.ndarray
            Ground truth values
        reference_model : str
            Name of the reference model (proposed AHSE)
        h : int
            Conservé pour rétrocompatibilité (non utilisé pour HAC)
        hac_nlags : int
            Lags Newey-West (96 pour 15-min, 24 pour 1h)

        Returns
        -------
        pd.DataFrame with DM statistics and p-values
        """
        results = []

        if reference_model not in predictions:
            raise ValueError(f"Reference model '{reference_model}' not in predictions")

        e_ref = y_true - predictions[reference_model]

        for model_name, preds in predictions.items():
            if model_name == reference_model:
                continue

            e_model = y_true - preds

            dm_stat, p_value = DieboldMarianoTest.compute_dm_statistic(
                e_model, e_ref, h=h, hac_nlags=hac_nlags
            )
            
            # Significance interpretation
            if p_value < 0.01:
                significance = '***'
            elif p_value < 0.05:
                significance = '**'
            elif p_value < 0.10:
                significance = '*'
            else:
                significance = ''
            
            results.append({
                'Model': model_name,
                'DM_Statistic': dm_stat,
                'P_Value': p_value,
                'Significance': significance,
                'AHSE_Better': dm_stat > 0
            })
        
        return pd.DataFrame(results)


class ResidualAnalyzer:
    """
    Comprehensive residual analysis for forecast evaluation.
    
    Provides:
    - Descriptive statistics (Mean, Std, Skewness, Kurtosis)
    - Normality tests (Kolmogorov-Smirnov)
    - Heteroscedasticity analysis
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = np.asarray(y_true).flatten()
        self.y_pred = np.asarray(y_pred).flatten()
        self.residuals = self.y_true - self.y_pred
    
    def compute_statistics(self) -> Dict[str, float]:
        """Compute descriptive statistics of residuals."""
        return {
            'Mean': np.mean(self.residuals),
            'Std': np.std(self.residuals),
            'Median': np.median(self.residuals),
            'Skewness': skew(self.residuals),
            'Kurtosis': kurtosis(self.residuals),
            'Min': np.min(self.residuals),
            'Max': np.max(self.residuals),
            'IQR': np.percentile(self.residuals, 75) - np.percentile(self.residuals, 25)
        }
    
    def normality_test(self) -> Dict[str, float]:
        """
        Perform Kolmogorov-Smirnov test for normality.
        
        H0: Residuals follow a normal distribution
        """
        # Standardize residuals
        standardized = (self.residuals - np.mean(self.residuals)) / np.std(self.residuals)
        
        # KS test against standard normal
        ks_stat, p_value = kstest(standardized, 'norm')
        
        return {
            'KS_Statistic': ks_stat,
            'P_Value': p_value,
            'Is_Normal': p_value > 0.05
        }
    
    def autocorrelation_test(self, max_lags: int = 10) -> Dict[str, np.ndarray]:
        """Compute autocorrelation of residuals."""
        n = len(self.residuals)
        acf = np.zeros(max_lags)
        
        for lag in range(max_lags):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.corrcoef(
                    self.residuals[:-lag], 
                    self.residuals[lag:]
                )[0, 1]
        
        # Ljung-Box Q statistic
        q_stat = n * (n + 2) * np.sum(acf[1:]**2 / (n - np.arange(1, max_lags)))
        
        return {
            'ACF': acf,
            'Ljung_Box_Q': q_stat,
            'Lags': np.arange(max_lags)
        }
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive residual analysis report."""
        stats = self.compute_statistics()
        normality = self.normality_test()
        
        report_data = {
            'Metric': list(stats.keys()) + ['KS_Stat', 'KS_P_Value', 'Normal_Distribution'],
            'Value': list(stats.values()) + [
                normality['KS_Statistic'],
                normality['P_Value'],
                'Yes' if normality['Is_Normal'] else 'No'
            ]
        }
        
        return pd.DataFrame(report_data)


class FeatureImportanceAnalyzer:
    """
    Feature importance analysis using tree-based model gains.
    Provides XAI (Explainable AI) for solar forecasting models.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize with a trained tree-based model.
        
        Parameters:
        -----------
        model : XGBoost or LightGBM model
        feature_names : List of feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    def extract_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Parameters:
        -----------
        importance_type : str
            'gain', 'weight', or 'cover' for XGBoost
            'gain' or 'split' for LightGBM
        """
        model_type = type(self.model).__name__
        
        if hasattr(self.model, 'feature_importances_'):
            # sklearn-style interface
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'get_booster'):
            # XGBoost
            booster = self.model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)
            importance = np.array([
                importance_dict.get(f'f{i}', 0) 
                for i in range(len(self.feature_names))
            ])
        elif hasattr(self.model, 'feature_importance'):
            # LightGBM
            importance = self.model.feature_importance(importance_type=importance_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Normalize to sum to 1
        importance = importance / np.sum(importance) if np.sum(importance) > 0 else importance
        
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return df
    
    def plot_importance(
        self, 
        top_k: int = 10, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot feature importance bar chart."""
        df = self.extract_importance().head(top_k)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        
        bars = ax.barh(
            df['Feature'], 
            df['Importance'],
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add value labels
        for bar, val in zip(bars, df['Importance']):
            ax.text(
                bar.get_width() + 0.005, 
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                va='center',
                fontsize=10
            )
        
        ax.set_xlabel('Normalized Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance for Solar Power Forecasting\n(XGBoost/LightGBM Gain)', 
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Highlight key meteorological features
        key_features = ['GHI', 'Temperature', 'zenith_angle', 'kt', 'clear_sky_ghi']
        for i, label in enumerate(ax.get_yticklabels()):
            if any(kf in label.get_text() for kf in key_features):
                label.set_fontweight('bold')
                label.set_color('#E63946')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
        
        return fig, ax


class TrainingHealthAnalyzer:
    """
    Analyze training dynamics to detect overfitting and convergence issues.
    """
    
    def __init__(
        self,
        train_losses: List[float],
        val_losses: List[float],
        model_name: str = 'Model'
    ):
        self.train_losses = np.array(train_losses)
        self.val_losses = np.array(val_losses)
        self.model_name = model_name
        self.epochs = np.arange(1, len(train_losses) + 1)
    
    def detect_overfitting(self, patience: int = 5) -> Dict[str, any]:
        """
        Detect overfitting based on train/val divergence.
        
        Returns:
        --------
        Dict with overfitting analysis results
        """
        # Find best validation epoch
        best_val_epoch = np.argmin(self.val_losses) + 1
        
        # Check if val loss increased after best epoch
        if best_val_epoch < len(self.val_losses):
            post_best_val = self.val_losses[best_val_epoch:]
            post_best_train = self.train_losses[best_val_epoch:]
            
            # Gap between train and val at end vs beginning
            initial_gap = self.val_losses[0] - self.train_losses[0]
            final_gap = self.val_losses[-1] - self.train_losses[-1]
            gap_increase = final_gap - initial_gap
            
            is_overfitting = gap_increase > 0.05 * np.mean(self.val_losses)
        else:
            gap_increase = 0
            is_overfitting = False
        
        return {
            'Best_Val_Epoch': best_val_epoch,
            'Best_Val_Loss': np.min(self.val_losses),
            'Final_Train_Loss': self.train_losses[-1],
            'Final_Val_Loss': self.val_losses[-1],
            'Train_Val_Gap': self.val_losses[-1] - self.train_losses[-1],
            'Gap_Increase': gap_increase,
            'Is_Overfitting': is_overfitting,
            'Early_Stopped': len(self.train_losses) < 100  # Assuming max 100 epochs
        }
    
    def convergence_quality(self) -> Dict[str, float]:
        """Assess quality of convergence."""
        # Compute smoothed gradient of validation loss
        window = min(5, len(self.val_losses) // 3)
        if window > 1:
            smoothed = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
            gradient = np.gradient(smoothed)
            final_gradient = np.mean(gradient[-window:])
        else:
            final_gradient = self.val_losses[-1] - self.val_losses[0]
        
        # Convergence ratio
        improvement = (self.val_losses[0] - np.min(self.val_losses)) / self.val_losses[0]
        
        return {
            'Total_Epochs': len(self.train_losses),
            'Final_Gradient': final_gradient,
            'Improvement_Ratio': improvement,
            'Converged': abs(final_gradient) < 0.001
        }
    
    def plot_learning_curves(
        self, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.epochs, self.train_losses, 'b-', 
                linewidth=2, label='Training Loss', alpha=0.8)
        ax.plot(self.epochs, self.val_losses, 'r-', 
                linewidth=2, label='Validation Loss', alpha=0.8)
        
        # Mark best validation epoch
        best_epoch = np.argmin(self.val_losses) + 1
        best_val = np.min(self.val_losses)
        ax.axvline(x=best_epoch, color='green', linestyle='--', 
                   alpha=0.7, label=f'Best Val (Epoch {best_epoch})')
        ax.scatter([best_epoch], [best_val], color='green', s=100, zorder=5)
        
        # Shade overfitting region if applicable
        analysis = self.detect_overfitting()
        if analysis['Is_Overfitting']:
            ax.axvspan(best_epoch, len(self.epochs), 
                      alpha=0.1, color='red', label='Overfitting Region')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training Dynamics: {self.model_name}\n'
                     f'Best Val Loss: {best_val:.4f} at Epoch {best_epoch}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add convergence annotation
        conv = self.convergence_quality()
        ax.text(0.02, 0.02, 
                f"Improvement: {conv['Improvement_Ratio']:.1%}\n"
                f"Converged: {'Yes' if conv['Converged'] else 'No'}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
        
        return fig, ax


def run_full_statistical_analysis(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    feature_importance_model=None,
    feature_names: List[str] = None,
    training_histories: Dict[str, Tuple[List[float], List[float]]] = None,
    output_dir: str = 'results/figures'
) -> Dict[str, pd.DataFrame]:
    """
    Run complete statistical analysis pipeline.
    
    Parameters:
    -----------
    predictions : Dict[str, np.ndarray]
        Model predictions
    y_true : np.ndarray
        Ground truth
    feature_importance_model : Optional model
        Trained XGBoost/LightGBM for feature importance
    feature_names : List[str]
        Feature names for importance analysis
    training_histories : Dict[str, Tuple]
        Training histories {model_name: (train_losses, val_losses)}
    output_dir : str
        Directory to save figures
        
    Returns:
    --------
    Dict with analysis results DataFrames
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS FOR THESIS")
    print("="*70)
    
    # 1. Diebold-Mariano Tests
    print("\n📊 1. Diebold-Mariano Statistical Significance Tests")
    print("-"*50)
    
    if 'AHSE' in predictions or 'AHSE (Proposed)' in predictions:
        ref_model = 'AHSE' if 'AHSE' in predictions else 'AHSE (Proposed)'
        dm_results = DieboldMarianoTest.run_pairwise_tests(
            predictions, y_true, reference_model=ref_model
        )
        results['diebold_mariano'] = dm_results
        print(dm_results.to_string(index=False))
        dm_results.to_csv(os.path.join(output_dir, '../tables/dm_test_results.csv'), index=False)
    
    # 2. Residual Analysis for each model
    print("\n📊 2. Residual Analysis")
    print("-"*50)
    
    residual_results = []
    for model_name, preds in predictions.items():
        analyzer = ResidualAnalyzer(y_true.flatten(), preds.flatten())
        stats = analyzer.compute_statistics()
        normality = analyzer.normality_test()
        
        residual_results.append({
            'Model': model_name,
            'Mean': stats['Mean'],
            'Std': stats['Std'],
            'Skewness': stats['Skewness'],
            'Kurtosis': stats['Kurtosis'],
            'KS_Stat': normality['KS_Statistic'],
            'KS_P_Value': normality['P_Value'],
            'Is_Normal': 'Yes' if normality['Is_Normal'] else 'No'
        })
    
    residual_df = pd.DataFrame(residual_results)
    results['residual_analysis'] = residual_df
    print(residual_df.to_string(index=False))
    residual_df.to_csv(os.path.join(output_dir, '../tables/residual_analysis.csv'), index=False)
    
    # 3. Feature Importance (if model provided)
    if feature_importance_model is not None and feature_names is not None:
        print("\n📊 3. Feature Importance (XAI)")
        print("-"*50)
        
        fi_analyzer = FeatureImportanceAnalyzer(feature_importance_model, feature_names)
        fi_df = fi_analyzer.extract_importance()
        results['feature_importance'] = fi_df
        print(fi_df.head(10).to_string(index=False))
        
        fi_analyzer.plot_importance(
            save_path=os.path.join(output_dir, 'fig14_feature_importance.png')
        )
        fi_df.to_csv(os.path.join(output_dir, '../tables/feature_importance.csv'), index=False)
    
    # 4. Training Health Analysis
    if training_histories is not None:
        print("\n📊 4. Training Health Analysis")
        print("-"*50)
        
        health_results = []
        for model_name, (train_losses, val_losses) in training_histories.items():
            analyzer = TrainingHealthAnalyzer(train_losses, val_losses, model_name)
            overfitting = analyzer.detect_overfitting()
            convergence = analyzer.convergence_quality()
            
            health_results.append({
                'Model': model_name,
                'Best_Epoch': overfitting['Best_Val_Epoch'],
                'Best_Val_Loss': overfitting['Best_Val_Loss'],
                'Is_Overfitting': 'Yes' if overfitting['Is_Overfitting'] else 'No',
                'Converged': 'Yes' if convergence['Converged'] else 'No',
                'Improvement': f"{convergence['Improvement_Ratio']:.1%}"
            })
            
            # Save individual learning curve
            analyzer.plot_learning_curves(
                save_path=os.path.join(output_dir, f'training_health_{model_name}.png')
            )
            plt.close()
        
        health_df = pd.DataFrame(health_results)
        results['training_health'] = health_df
        print(health_df.to_string(index=False))
        health_df.to_csv(os.path.join(output_dir, '../tables/training_health.csv'), index=False)
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    print("Statistical Analysis Module loaded successfully.")
    print("Use run_full_statistical_analysis() with your actual predictions and ground truth.")
    print("\nExample usage:")
    print("  from src.evaluation.statistical_analysis import run_full_statistical_analysis")
    print("  results = run_full_statistical_analysis(predictions, y_true, output_dir='results/figures')")
