"""
Training Module
===============
Entraînement des modèles de prévision PV.
"""

from .trainer import (
    MultiHorizonTrainer,
    TreeModelTrainer,
    ModelTrainer  # Alias
)
from .loss_functions import (
    MSELoss,
    MAELoss,
    HuberLoss,
    PinballLoss,
    MultiQuantileLoss,
    WeightedMSELoss,
    SkillScoreLoss,
    DaylightMaskedLoss,
    get_loss_function
)

__all__ = [
    'MultiHorizonTrainer',
    'TreeModelTrainer',
    'ModelTrainer',
    'MSELoss',
    'MAELoss',
    'HuberLoss',
    'PinballLoss',
    'MultiQuantileLoss',
    'WeightedMSELoss',
    'SkillScoreLoss',
    'DaylightMaskedLoss',
    'get_loss_function'
]
