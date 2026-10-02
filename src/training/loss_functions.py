"""
Fonctions de Perte pour Prévision PV
====================================
MSE standard et Pinball Loss pour prévision probabiliste.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MSELoss(nn.Module):
    """MSE Loss standard."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


class MAELoss(nn.Module):
    """MAE Loss (L1)."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


class HuberLoss(nn.Module):
    """Huber Loss (smooth L1) - robuste aux outliers."""
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.delta, reduction=self.reduction)


class PinballLoss(nn.Module):
    """
    Pinball Loss pour prévision probabiliste (quantile regression).
    
    Permet d'estimer des intervalles de confiance en entraînant
    le modèle sur différents quantiles (ex: 10%, 50%, 90%).
    
    Loss = max(q * (y - pred), (q - 1) * (y - pred))
    
    Args:
        quantile: Quantile cible (0 < q < 1)
    """
    
    def __init__(self, quantile: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        assert 0 < quantile < 1, "Quantile doit être entre 0 et 1"
        self.quantile = quantile
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - pred
        loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiQuantileLoss(nn.Module):
    """
    Perte pour prévision multi-quantile.
    
    Entraîne simultanément sur plusieurs quantiles pour
    générer des intervalles de prédiction.
    
    Args:
        quantiles: Liste des quantiles (ex: [0.1, 0.5, 0.9])
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.losses = nn.ModuleList([
            PinballLoss(q) for q in quantiles
        ])
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (batch, horizon, n_quantiles) ou (batch, horizon * n_quantiles)
            target: (batch, horizon)
        """
        if pred.dim() == 2:
            # Reshape si nécessaire
            batch, total = pred.shape
            horizon = total // self.n_quantiles
            pred = pred.view(batch, horizon, self.n_quantiles)
        
        total_loss = 0
        for i, loss_fn in enumerate(self.losses):
            pred_q = pred[:, :, i]
            total_loss = total_loss + loss_fn(pred_q, target)
        
        return total_loss / self.n_quantiles


class WeightedMSELoss(nn.Module):
    """
    MSE pondéré par l'horizon.
    
    Donne plus d'importance aux premiers pas de temps
    (plus faciles à prédire) ou aux derniers (plus importants).
    """
    
    def __init__(
        self, 
        horizon: int, 
        decay: float = 0.9,
        increasing: bool = False
    ):
        super().__init__()
        
        if increasing:
            # Plus de poids sur les horizons lointains
            weights = torch.tensor([decay ** (horizon - 1 - i) for i in range(horizon)])
        else:
            # Plus de poids sur les horizons proches
            weights = torch.tensor([decay ** i for i in range(horizon)])
        
        weights = weights / weights.sum()
        self.register_buffer('weights', weights)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # (batch, horizon)
        mse = (pred - target) ** 2
        weighted_mse = mse * self.weights.unsqueeze(0)
        return weighted_mse.mean()


class SkillScoreLoss(nn.Module):
    """
    Perte basée sur le Skill Score par rapport à la persistence.
    
    Encourage le modèle à faire mieux que simplement répéter
    la dernière valeur observée.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Pondération entre MSE et skill score (0 = MSE pur, 1 = skill pur)
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        persistence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Prédictions du modèle
            target: Valeurs réelles
            persistence: Prédictions du modèle de persistence (optionnel)
        """
        mse_loss = self.mse(pred, target)
        
        if persistence is None:
            return mse_loss
        
        # Skill score: 1 - MSE_model / MSE_persistence
        mse_persistence = self.mse(persistence, target)
        skill = 1 - mse_loss / (mse_persistence + 1e-8)
        
        # Combiner: minimiser MSE et maximiser skill (donc minimiser -skill)
        return (1 - self.alpha) * mse_loss - self.alpha * skill


class DaylightMaskedLoss(nn.Module):
    """
    Perte masquée pour ignorer les données nocturnes.
    
    Calcule la perte uniquement sur les points où GHI > seuil.
    """
    
    def __init__(
        self, 
        base_loss: nn.Module = None,
        threshold: float = 10.0
    ):
        super().__init__()
        self.base_loss = base_loss or nn.MSELoss(reduction='none')
        self.threshold = threshold
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        ghi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Prédictions
            target: Valeurs réelles
            ghi: Valeurs GHI correspondantes pour le masquage
        """
        losses = self.base_loss(pred, target)
        
        if ghi is not None:
            mask = (ghi > self.threshold).float()
            n_valid = mask.sum()
            
            if n_valid > 0:
                return (losses * mask).sum() / n_valid
            else:
                return losses.mean()
        
        return losses.mean()


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Factory pour créer une fonction de perte.
    
    Args:
        name: "mse", "mae", "huber", "pinball", "multi_quantile"
    """
    name = name.lower()
    
    if name == "mse":
        return MSELoss(**kwargs)
    elif name == "mae":
        return MAELoss(**kwargs)
    elif name == "huber":
        return HuberLoss(**kwargs)
    elif name == "pinball":
        return PinballLoss(**kwargs)
    elif name == "multi_quantile":
        return MultiQuantileLoss(**kwargs)
    elif name == "weighted_mse":
        return WeightedMSELoss(**kwargs)
    else:
        raise ValueError(f"Loss inconnue: {name}")
