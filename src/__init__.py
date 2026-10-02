"""
PV Forecasting SOTA Package
============================
State-of-the-art PV power forecasting system for Master's thesis research.

University: Wuhan University
Department: Electrical Engineering

Key Features:
- Clear Sky Index (kt) normalization via pvlib
- Multi-horizon direct forecasting (H=96 for 24h at 15min)
- SOTA models: PatchTST, N-HiTS
- Daytime-masked evaluation for scientific rigor
- Weather regime analysis

Modules:
--------
data_engine : Data loading, preprocessing, Clear Sky calculations
models      : PatchTST, N-HiTS, LSTM, GRU, Transformer, Ensembles
training    : Multi-horizon trainers, loss functions
evaluation  : Metrics factory, horizon degradation analysis
visualization : Publication-quality plots

Usage:
------
>>> from src.data_engine import load_and_prepare_multi_horizon, ClearSkyCalculator
>>> from src.models import create_patchtst_from_config, create_nhits_from_config
>>> from src.training import MultiHorizonTrainer
>>> from src.evaluation import MetricsFactory, HorizonErrorAnalyzer
>>> from src.visualization import plot_forecast_comparison

Example:
--------
>>> # Load data with kt normalization
>>> data = load_and_prepare_multi_horizon(
...     "data/processed/data_15min.csv",
...     config_path="configs/data_config_yulara_neighbours.yaml"
... )
>>> 
>>> # Create SOTA model
>>> model = create_patchtst_from_config(
...     n_features=data['n_features'],
...     seq_len=96, pred_len=96
... )
>>> 
>>> # Train
>>> trainer = MultiHorizonTrainer()
>>> model, history = trainer.train(model, data)
>>> 
>>> # Evaluate with daytime masking
>>> metrics = MetricsFactory.compute_all(
...     y_true=data['y_test'],
...     y_pred=predictions,
...     daytime_mask=data['daytime_mask']
... )
"""

__version__ = '2.0.0'
__author__ = 'Master Student - Wuhan University'

# High-level imports for convenience
from .data_engine import (
    ClearSkyCalculator,
    MultiHorizonPreprocessor,
    load_and_prepare_multi_horizon
)

from .models import (
    PatchTST,
    NHiTS,
    LSTMMultiHorizon,
    GRUMultiHorizon,
    TransformerMultiHorizon,
    AHSEMultiHorizon,
    create_patchtst_from_config,
    create_nhits_from_config
)

from .training import (
    MultiHorizonTrainer,
    TreeModelTrainer,
    get_loss_function
)

from .evaluation import (
    MetricsFactory,
    MetricsCalculator,
    HorizonDegradationAnalyzer,
    HorizonErrorAnalyzer,
    compute_skill_score
)

from .visualization import (
    plot_forecast_comparison,
    plot_horizon_degradation,
    plot_model_comparison_heatmap,
    save_figure_for_publication
)

__all__ = [
    # Version
    '__version__',
    # Data
    'ClearSkyCalculator',
    'MultiHorizonPreprocessor',
    'load_and_prepare_multi_horizon',
    # Models
    'PatchTST',
    'NHiTS',
    'LSTMMultiHorizon',
    'GRUMultiHorizon',
    'TransformerMultiHorizon',
    'AHSEMultiHorizon',
    'create_patchtst_from_config',
    'create_nhits_from_config',
    # Training
    'MultiHorizonTrainer',
    'TreeModelTrainer',
    'get_loss_function',
    # Evaluation
    'MetricsFactory',
    'MetricsCalculator',
    'HorizonDegradationAnalyzer',
    'HorizonErrorAnalyzer',
    'compute_skill_score',
    # Visualization
    'plot_forecast_comparison',
    'plot_horizon_degradation',
    'plot_model_comparison_heatmap',
    'save_figure_for_publication'
]
