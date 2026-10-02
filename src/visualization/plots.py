"""
Publication-Quality Plotting Module
====================================
Figures conformes aux standards IEEE/Elsevier pour thèse de Master.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configuration style publication
plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class PlotConfig:
    """Configuration pour figures de publication."""
    # Taille figure (IEEE column width = 3.5in, full width = 7in)
    figsize_single: Tuple[float, float] = (3.5, 2.8)
    figsize_double: Tuple[float, float] = (7.0, 3.5)
    figsize_full: Tuple[float, float] = (7.0, 5.5)
    
    # Police
    fontsize_title: int = 11
    fontsize_label: int = 10
    fontsize_tick: int = 9
    fontsize_legend: int = 8
    font_family: str = 'serif'
    
    # Couleurs (palette colorblind-friendly)
    colors: Dict[str, str] = None
    
    # DPI pour export
    dpi: int = 300
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'PatchTST': '#0072B2',      # Bleu
                'N-HiTS': '#D55E00',         # Orange
                'LSTM': '#009E73',           # Vert
                'GRU': '#CC79A7',            # Rose
                'Transformer': '#F0E442',    # Jaune
                'LightGBM': '#56B4E9',       # Bleu clair
                'XGBoost': '#E69F00',        # Orange clair
                'AHSE': '#000000',           # Noir
                'Persistence': '#999999',    # Gris
                'SmartPersistence': '#666666',
                'actual': '#000000',
                'clear_sky': '#FFD700',
            }
    
    def apply_style(self):
        """Applique le style de publication."""
        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': self.fontsize_tick,
            'axes.titlesize': self.fontsize_title,
            'axes.labelsize': self.fontsize_label,
            'xtick.labelsize': self.fontsize_tick,
            'ytick.labelsize': self.fontsize_tick,
            'legend.fontsize': self.fontsize_legend,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })


class ForecastPlotter:
    """Visualisation des prévisions multi-horizon."""
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.config.apply_style()
    
    def plot_forecast_vs_actual(
        self,
        timestamps: pd.DatetimeIndex,
        y_actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        clear_sky: Optional[np.ndarray] = None,
        title: str = "Prévision PV Multi-Horizon",
        ylabel: str = "Puissance (W)",
        show_confidence: bool = False,
        highlight_days: int = 3,
        ax: plt.Axes = None
    ) -> plt.Figure:
        """
        Trace les prévisions vs valeurs réelles.
        
        Args:
            timestamps: Index temporel
            y_actual: Valeurs réelles (n_samples,) ou (n_samples, horizon)
            predictions: Dict {model_name: predictions}
            clear_sky: Puissance ciel clair optionnelle
            title: Titre du graphique
            ylabel: Label axe Y
            show_confidence: Afficher intervalles de confiance
            highlight_days: Nombre de jours à afficher
            ax: Axes matplotlib existants
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize_double)
        else:
            fig = ax.figure
        
        # Limiter aux derniers jours
        n_points = min(len(timestamps), highlight_days * 96)  # 96 = 24h à 15min
        timestamps = timestamps[-n_points:]
        y_actual = np.array(y_actual).flatten()[-n_points:]
        
        # Tracer valeurs réelles
        ax.plot(timestamps, y_actual, 'k-', linewidth=1.5, 
                label='Réel', alpha=0.9, zorder=10)
        
        # Clear sky si disponible
        if clear_sky is not None:
            clear_sky = np.array(clear_sky).flatten()[-n_points:]
            ax.fill_between(timestamps, 0, clear_sky, 
                          alpha=0.2, color=self.config.colors['clear_sky'],
                          label='Ciel clair')
        
        # Tracer prédictions
        for model_name, preds in predictions.items():
            preds = np.array(preds).flatten()[-n_points:]
            color = self.config.colors.get(model_name, '#333333')
            ax.plot(timestamps, preds, '-', linewidth=1.0, 
                   label=model_name, color=color, alpha=0.8)
        
        # Formatage
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper left', framealpha=0.9, ncol=2)
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_multi_horizon_forecast(
        self,
        y_actual: np.ndarray,
        y_pred: np.ndarray,
        horizon_labels: List[str] = None,
        model_name: str = "Model",
        sample_idx: int = 0
    ) -> plt.Figure:
        """
        Trace une prévision multi-horizon pour un échantillon.
        
        Args:
            y_actual: Shape (n_samples, horizon)
            y_pred: Shape (n_samples, horizon)
            horizon_labels: Labels pour chaque pas
            model_name: Nom du modèle
            sample_idx: Index de l'échantillon à afficher
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_single)
        
        horizon = y_actual.shape[1] if y_actual.ndim > 1 else len(y_actual)
        x = np.arange(horizon)
        
        actual = y_actual[sample_idx] if y_actual.ndim > 1 else y_actual
        pred = y_pred[sample_idx] if y_pred.ndim > 1 else y_pred
        
        ax.plot(x, actual, 'k-', linewidth=1.5, marker='o', 
                markersize=3, label='Réel')
        ax.plot(x, pred, '-', linewidth=1.0, marker='s', 
                markersize=2, label=model_name,
                color=self.config.colors.get(model_name, '#0072B2'))
        
        if horizon_labels:
            step = max(1, horizon // 8)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(horizon_labels[::step], rotation=45)
        else:
            ax.set_xlabel('Horizon (pas)')
        
        ax.set_ylabel('Puissance (W)')
        ax.set_title(f'Prévision {model_name} - Échantillon #{sample_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class MetricsPlotter:
    """Visualisation des métriques de performance."""
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.config.apply_style()
    
    def plot_model_comparison_bar(
        self,
        metrics_df: pd.DataFrame,
        metric: str = 'RMSE',
        title: str = None,
        sort: bool = True
    ) -> plt.Figure:
        """
        Barplot de comparaison des modèles.
        
        Args:
            metrics_df: DataFrame avec colonnes ['Model', metric, ...]
            metric: Métrique à afficher
            title: Titre du graphique
            sort: Trier par performance
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_single)
        
        df = metrics_df.copy()
        if sort:
            ascending = metric not in ('R2', 'R²')
            df = df.sort_values(metric, ascending=ascending)
        
        colors = [self.config.colors.get(m, '#333333') for m in df['Model']]
        
        bars = ax.barh(df['Model'], df[metric], color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
        
        # Ajouter valeurs
        for bar, val in zip(bars, df[metric]):
            width = bar.get_width()
            ax.text(width + 0.01 * df[metric].max(), bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=self.config.fontsize_tick)
        
        ax.set_xlabel(metric)
        ax.set_title(title or f'Comparaison des modèles - {metric}')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_heatmap(
        self,
        metrics_df: pd.DataFrame,
        metrics: List[str] = ['RMSE', 'MAE', 'R2'],
        title: str = "Performance des Modèles"
    ) -> plt.Figure:
        """
        Heatmap des métriques pour tous les modèles.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_single)
        
        # Préparer données
        df = metrics_df.set_index('Model')[metrics].copy()
        
        # Normaliser pour visualisation (0-1)
        df_norm = df.copy()
        for col in df_norm.columns:
            if col in ('R2', 'R²'):
                # R² : plus élevé = mieux
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            else:
                # RMSE, MAE : plus bas = mieux (inverser)
                df_norm[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
        
        # Créer heatmap
        im = ax.imshow(df_norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df.index)
        
        # Annoter avec valeurs réelles
        for i in range(len(df)):
            for j in range(len(metrics)):
                val = df.iloc[i, j]
                text = f'{val:.3f}' if val < 1 else f'{val:.1f}'
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=self.config.fontsize_tick,
                       color='white' if df_norm.iloc[i, j] < 0.4 or df_norm.iloc[i, j] > 0.6 else 'black')
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Performance (normalisée)')
        
        plt.tight_layout()
        return fig
    
    def plot_skill_score_radar(
        self,
        skill_scores: Dict[str, Dict[str, float]],
        title: str = "Skill Score par Horizon"
    ) -> plt.Figure:
        """
        Radar plot des skill scores.
        
        Args:
            skill_scores: {model: {horizon: score}}
        """
        models = list(skill_scores.keys())
        horizons = list(skill_scores[models[0]].keys())
        n_horizons = len(horizons)
        
        # Angles pour le radar
        angles = np.linspace(0, 2 * np.pi, n_horizons, endpoint=False).tolist()
        angles += angles[:1]  # Fermer le cercle
        
        fig, ax = plt.subplots(figsize=self.config.figsize_single, 
                               subplot_kw=dict(polar=True))
        
        for model in models:
            values = [skill_scores[model][h] for h in horizons]
            values += values[:1]
            color = self.config.colors.get(model, '#333333')
            ax.plot(angles, values, 'o-', linewidth=1.5, 
                   label=model, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(horizons, fontsize=self.config.fontsize_tick)
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        return fig


class HorizonAnalysisPlotter:
    """Analyse de la dégradation des erreurs selon l'horizon."""
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.config.apply_style()
    
    def plot_error_by_horizon(
        self,
        horizon_errors: Dict[str, np.ndarray],
        metric: str = 'RMSE',
        resolution_minutes: int = 15,
        title: str = None
    ) -> plt.Figure:
        """
        Trace la dégradation de l'erreur selon l'horizon.
        
        Args:
            horizon_errors: {model: errors_per_horizon} shape (horizon,)
            metric: Nom de la métrique
            resolution_minutes: Résolution temporelle
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_double)
        
        first_model = list(horizon_errors.keys())[0]
        n_horizons = len(horizon_errors[first_model])
        hours = np.arange(n_horizons) * resolution_minutes / 60
        
        for model, errors in horizon_errors.items():
            color = self.config.colors.get(model, '#333333')
            ax.plot(hours, errors, '-', linewidth=1.5, 
                   label=model, color=color, marker='o', 
                   markersize=3, markevery=max(1, n_horizons//10))
        
        ax.set_xlabel('Horizon de Prévision (heures)')
        ax.set_ylabel(metric)
        ax.set_title(title or f'Dégradation {metric} vs Horizon')
        ax.legend(loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Zones colorées pour horizons
        if n_horizons >= 24:
            ax.axvspan(0, 1, alpha=0.1, color='green', label='Court terme')
            ax.axvspan(1, 6, alpha=0.1, color='yellow')
            ax.axvspan(6, hours[-1], alpha=0.1, color='red')
        
        plt.tight_layout()
        return fig
    
    def plot_horizon_skill_score(
        self,
        model_errors: Dict[str, np.ndarray],
        persistence_errors: np.ndarray,
        resolution_minutes: int = 15,
        title: str = "Skill Score par Horizon"
    ) -> plt.Figure:
        """
        Trace le Skill Score relatif à la persistance.
        
        Args:
            model_errors: {model: rmse_per_horizon}
            persistence_errors: RMSE de la persistance par horizon
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_double)
        
        n_horizons = len(persistence_errors)
        hours = np.arange(n_horizons) * resolution_minutes / 60
        
        # Ligne de référence (Persistence = 0)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        for model, errors in model_errors.items():
            skill = 1 - errors / (persistence_errors + 1e-8)
            skill = np.clip(skill, -1, 1)  # Limiter
            color = self.config.colors.get(model, '#333333')
            ax.plot(hours, skill, '-', linewidth=1.5, 
                   label=model, color=color)
        
        ax.set_xlabel('Horizon de Prévision (heures)')
        ax.set_ylabel('Skill Score')
        ax.set_title(title)
        ax.legend(loc='lower left', ncol=2)
        ax.set_ylim(-0.5, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Zone positive/négative
        ax.fill_between(hours, 0, 1, alpha=0.1, color='green')
        ax.fill_between(hours, -0.5, 0, alpha=0.1, color='red')
        
        plt.tight_layout()
        return fig


class WeatherRegimePlotter:
    """Visualisation des performances par régime météo."""
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.config.apply_style()
    
    def plot_regime_performance(
        self,
        regime_metrics: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = 'R2',
        title: str = "Performance par Régime Météo"
    ) -> plt.Figure:
        """
        Barplot groupé par régime météo.
        
        Args:
            regime_metrics: {regime: {model: {metric: value}}}
            metric: Métrique à afficher
        """
        fig, ax = plt.subplots(figsize=self.config.figsize_double)
        
        regimes = list(regime_metrics.keys())
        models = list(regime_metrics[regimes[0]].keys())
        n_regimes = len(regimes)
        n_models = len(models)
        
        x = np.arange(n_regimes)
        width = 0.8 / n_models
        
        for i, model in enumerate(models):
            values = [regime_metrics[r].get(model, {}).get(metric, 0) for r in regimes]
            offset = (i - n_models/2 + 0.5) * width
            color = self.config.colors.get(model, '#333333')
            ax.bar(x + offset, values, width, label=model, color=color, alpha=0.8)
        
        ax.set_xlabel('Régime Météorologique')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace('_', ' ').title() for r in regimes])
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_regime_distribution(
        self,
        regime_counts: Dict[str, int],
        title: str = "Distribution des Régimes Météo"
    ) -> plt.Figure:
        """Pie chart de la distribution des régimes."""
        fig, ax = plt.subplots(figsize=self.config.figsize_single)
        
        labels = [r.replace('_', ' ').title() for r in regime_counts.keys()]
        sizes = list(regime_counts.values())
        colors_regime = ['#FFD700', '#87CEEB', '#A9A9A9', '#98FB98']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, 
                                          colors=colors_regime[:len(labels)],
                                          autopct='%1.1f%%',
                                          startangle=90)
        
        ax.set_title(title)
        
        plt.tight_layout()
        return fig


# ============================================================================
# FONCTIONS DE COMMODITÉ
# ============================================================================

def create_publication_figure(
    figsize: str = 'single',
    config: PlotConfig = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Crée une figure avec style publication.
    
    Args:
        figsize: 'single', 'double', ou 'full'
        config: Configuration personnalisée
    """
    config = config or PlotConfig()
    config.apply_style()
    
    sizes = {
        'single': config.figsize_single,
        'double': config.figsize_double,
        'full': config.figsize_full
    }
    
    fig, ax = plt.subplots(figsize=sizes.get(figsize, config.figsize_single))
    return fig, ax


def plot_forecast_comparison(
    timestamps: pd.DatetimeIndex,
    y_actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    **kwargs
) -> plt.Figure:
    """Wrapper pour ForecastPlotter.plot_forecast_vs_actual()"""
    plotter = ForecastPlotter()
    return plotter.plot_forecast_vs_actual(timestamps, y_actual, predictions, **kwargs)


def plot_horizon_degradation(
    horizon_errors: Dict[str, np.ndarray],
    **kwargs
) -> plt.Figure:
    """Wrapper pour HorizonAnalysisPlotter.plot_error_by_horizon()"""
    plotter = HorizonAnalysisPlotter()
    return plotter.plot_error_by_horizon(horizon_errors, **kwargs)


def plot_model_comparison_heatmap(
    metrics_df: pd.DataFrame,
    **kwargs
) -> plt.Figure:
    """Wrapper pour MetricsPlotter.plot_metrics_heatmap()"""
    plotter = MetricsPlotter()
    return plotter.plot_metrics_heatmap(metrics_df, **kwargs)


def plot_weather_regime_performance(
    regime_metrics: Dict,
    **kwargs
) -> plt.Figure:
    """Wrapper pour WeatherRegimePlotter.plot_regime_performance()"""
    plotter = WeatherRegimePlotter()
    return plotter.plot_regime_performance(regime_metrics, **kwargs)


def plot_clear_sky_analysis(
    timestamps: pd.DatetimeIndex,
    actual_power: np.ndarray,
    clear_sky_power: np.ndarray,
    kt: np.ndarray,
    title: str = "Analyse Clear Sky Index"
) -> plt.Figure:
    """
    Analyse du Clear Sky Index (kt).
    """
    config = PlotConfig()
    config.apply_style()
    
    fig, axes = plt.subplots(3, 1, figsize=config.figsize_full, sharex=True)
    
    # Puissance
    ax1 = axes[0]
    ax1.plot(timestamps, actual_power, 'b-', linewidth=0.8, label='Réel', alpha=0.8)
    ax1.plot(timestamps, clear_sky_power, 'orange', linewidth=1.0, 
             label='Ciel clair', linestyle='--')
    ax1.set_ylabel('Puissance (W)')
    ax1.legend(loc='upper right')
    ax1.set_title('Puissance PV vs Ciel Clair')
    
    # kt
    ax2 = axes[1]
    ax2.plot(timestamps, kt, 'g-', linewidth=0.8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='kt=1')
    ax2.set_ylabel('Clear Sky Index (kt)')
    ax2.set_ylim(0, 1.5)
    ax2.legend(loc='upper right')
    ax2.set_title('Clear Sky Index')
    
    # Distribution kt
    ax3 = axes[2]
    kt_valid = kt[(kt > 0) & (kt <= 1.5)]
    ax3.hist(kt_valid, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Clear Sky Index (kt)')
    ax3.set_ylabel('Fréquence')
    ax3.set_title('Distribution de kt (période diurne)')
    
    fig.suptitle(title, fontsize=config.fontsize_title + 2, y=1.02)
    plt.tight_layout()
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    title: str = None
) -> plt.Figure:
    """
    Trace l'historique d'entraînement.
    
    Args:
        history: {'train_loss': [...], 'val_loss': [...]}
        model_name: Nom du modèle
    """
    config = PlotConfig()
    config.apply_style()
    
    fig, ax = plt.subplots(figsize=config.figsize_single)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=1.5, 
           label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=1.5, 
           label='Val Loss')
    
    # Marquer le meilleur
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
    ax.annotate(f'Best: {best_val:.4f}', 
                xy=(best_epoch, best_val),
                xytext=(best_epoch + len(epochs)*0.1, best_val),
                fontsize=config.fontsize_tick,
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title or f'Entraînement {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_residual_analysis(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    timestamps: pd.DatetimeIndex = None
) -> plt.Figure:
    """
    Analyse des résidus (erreurs de prédiction).
    """
    config = PlotConfig()
    config.apply_style()
    
    residuals = y_actual.flatten() - y_pred.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=config.figsize_full)
    
    # 1. Résidus vs temps
    ax1 = axes[0, 0]
    if timestamps is not None:
        ax1.plot(timestamps, residuals, 'b-', linewidth=0.5, alpha=0.7)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    else:
        ax1.plot(residuals, 'b-', linewidth=0.5, alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_ylabel('Résidu (W)')
    ax1.set_title('Résidus vs Temps')
    
    # 2. Histogramme des résidus
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=50, color='steelblue', alpha=0.7, 
             edgecolor='black', density=True)
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.axvline(x=np.mean(residuals), color='green', linestyle='-', 
               label=f'Moyenne: {np.mean(residuals):.1f}')
    ax2.set_xlabel('Résidu (W)')
    ax2.set_ylabel('Densité')
    ax2.set_title('Distribution des Résidus')
    ax2.legend()
    
    # 3. Résidus vs Prédit
    ax3 = axes[1, 0]
    ax3.scatter(y_pred.flatten(), residuals, alpha=0.3, s=5, c='steelblue')
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Valeur Prédite (W)')
    ax3.set_ylabel('Résidu (W)')
    ax3.set_title('Résidus vs Prédictions')
    
    # 4. QQ Plot
    ax4 = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normalité)')
    
    fig.suptitle(f'Analyse des Résidus - {model_name}', 
                 fontsize=config.fontsize_title + 2, y=1.02)
    plt.tight_layout()
    
    return fig


def save_figure_for_publication(
    fig: plt.Figure,
    filename: str,
    output_dir: str = 'results/figures',
    formats: List[str] = ['pdf', 'png'],
    dpi: int = 300
):
    """
    Sauvegarde une figure dans plusieurs formats.
    
    Args:
        fig: Figure matplotlib
        filename: Nom de fichier (sans extension)
        output_dir: Répertoire de sortie
        formats: Formats d'export
        dpi: Résolution
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, format=fmt, dpi=dpi, 
                    bbox_inches='tight', pad_inches=0.05)
        print(f"  ✓ Saved: {filepath}")
