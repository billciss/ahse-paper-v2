"""
Visualization Module
====================
Publication-quality plots for PV forecasting research.
"""

from .plots import (
    PlotConfig,
    ForecastPlotter,
    MetricsPlotter,
    HorizonAnalysisPlotter,
    WeatherRegimePlotter,
    create_publication_figure,
    plot_forecast_comparison,
    plot_horizon_degradation,
    plot_model_comparison_heatmap,
    plot_weather_regime_performance,
    plot_clear_sky_analysis,
    plot_training_history,
    plot_residual_analysis,
    save_figure_for_publication
)

__all__ = [
    'PlotConfig',
    'ForecastPlotter',
    'MetricsPlotter',
    'HorizonAnalysisPlotter',
    'WeatherRegimePlotter',
    'create_publication_figure',
    'plot_forecast_comparison',
    'plot_horizon_degradation',
    'plot_model_comparison_heatmap',
    'plot_weather_regime_performance',
    'plot_clear_sky_analysis',
    'plot_training_history',
    'plot_residual_analysis',
    'save_figure_for_publication'
]
