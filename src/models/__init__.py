"""
Models Module
=============
Modèles de prévision PV: SOTA et baselines.
"""

from .patch_tst import PatchTST, PatchTSTForecaster, create_patchtst_from_config
from .nhits import NHiTS, NHiTSForecaster, create_nhits_from_config
from .baselines import (
    LSTMMultiHorizon, 
    GRUMultiHorizon, 
    TransformerMultiHorizon,
    PersistenceModel,
    SmartPersistenceModel,
    create_model_from_config,
    # Alias
    LSTMModel,
    GRUModel,
    TransformerModel
)
from .ensembles import (
    AHSEMultiHorizon,
    AHSESelector,
    GatingNetwork,
    analyze_model_performance_by_regime
)

__all__ = [
    # SOTA Models
    'PatchTST',
    'PatchTSTForecaster',
    'create_patchtst_from_config',
    'NHiTS',
    'NHiTSForecaster',
    'create_nhits_from_config',
    # Baselines
    'LSTMMultiHorizon',
    'GRUMultiHorizon',
    'TransformerMultiHorizon',
    'PersistenceModel',
    'SmartPersistenceModel',
    'create_model_from_config',
    # Aliases
    'LSTMModel',
    'GRUModel',
    'TransformerModel',
    # Ensembles
    'AHSEMultiHorizon',
    'AHSESelector',
    'GatingNetwork',
    'analyze_model_performance_by_regime'
]
