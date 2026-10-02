"""
N-HiTS - Neural Hierarchical Interpolation for Time Series
==========================================================
Implémentation du modèle N-HiTS pour prévision multi-horizon.

Référence: Challu et al., "N-HiTS: Neural Hierarchical Interpolation 
           for Time Series Forecasting" (AAAI 2023)

Caractéristiques clés:
1. Multi-Rate Sampling: Capture différentes fréquences temporelles
2. Hierarchical Interpolation: Combine les prédictions à différentes résolutions
3. Expressivity Ratios: Contrôle la contribution de chaque stack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NHiTSBlock(nn.Module):
    """
    Bloc individuel de N-HiTS.
    
    Chaque bloc:
    1. Downsample l'entrée (max pooling)
    2. Passe par un MLP
    3. Génère des coefficients pour l'interpolation
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: List[int],
        pooling_size: int,
        n_theta: int,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.pooling_size = pooling_size
        self.n_theta = n_theta
        self.output_size = output_size
        
        # Pooling pour le downsampling
        self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)
        
        # Taille après pooling
        pooled_size = input_size // pooling_size
        
        # MLP
        layers = []
        in_size = pooled_size
        
        for units in hidden_units:
            layers.append(nn.Linear(in_size, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_size = units
        
        self.mlp = nn.Sequential(*layers)
        
        # Projection vers les coefficients theta
        # theta pour le backcast et le forecast
        self.theta_backcast = nn.Linear(in_size, n_theta)
        self.theta_forecast = nn.Linear(in_size, n_theta)
        
        # Bases pour l'interpolation
        self.backcast_basis = nn.Linear(n_theta, input_size, bias=False)
        self.forecast_basis = nn.Linear(n_theta, output_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_size)
        
        Returns:
            backcast: (batch, input_size) - reconstruction de l'entrée
            forecast: (batch, output_size) - prédiction
        """
        # Pooling: (batch, input_size) → (batch, 1, input_size) → pool → squeeze
        x_pooled = self.pooling(x.unsqueeze(1)).squeeze(1)
        
        # MLP
        h = self.mlp(x_pooled)
        
        # Coefficients theta
        theta_b = self.theta_backcast(h)
        theta_f = self.theta_forecast(h)
        
        # Interpolation
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)
        
        return backcast, forecast


class NHiTSStack(nn.Module):
    """
    Stack de blocs N-HiTS.
    
    Chaque stack a une résolution temporelle différente
    (via le pooling_size).
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_blocks: int,
        hidden_units: List[int],
        pooling_size: int,
        n_theta: int,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                input_size=input_size,
                output_size=output_size,
                hidden_units=hidden_units,
                pooling_size=pooling_size,
                n_theta=n_theta,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm
            )
            for _ in range(n_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_size)
        
        Returns:
            backcast: Résidu total
            forecast: Somme des forecasts
        """
        residual = x
        forecast_total = 0
        
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_total = forecast_total + forecast
        
        return residual, forecast_total


class NHiTS(nn.Module):
    """
    N-HiTS - Neural Hierarchical Interpolation for Time Series.
    
    Architecture:
    1. Input: Séquence temporelle aplatie
    2. Multiple Stacks: Chacun avec un pooling différent
    3. Output: Somme des forecasts de tous les stacks
    
    Args:
        input_size: Longueur de la séquence d'entrée × nombre de features
        output_size: Horizon de prédiction (H)
        n_blocks: Liste du nombre de blocs par stack
        mlp_units: Liste des dimensions MLP par stack
        pooling_sizes: Liste des tailles de pooling par stack
        n_theta: Nombre de coefficients d'interpolation
        dropout: Taux de dropout
        activation: Fonction d'activation
        batch_norm: Utiliser BatchNorm
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_blocks: List[int] = [3, 3, 3],
        mlp_units: List[List[int]] = [[512, 512], [512, 512], [512, 512]],
        pooling_sizes: List[int] = [4, 2, 1],
        n_theta: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        n_stacks = len(n_blocks)
        
        # n_theta par défaut
        if n_theta is None:
            n_theta = output_size
        
        logger.info(f"🔧 N-HiTS initialisé:")
        logger.info(f"   Input: {input_size} → Output: {output_size}")
        logger.info(f"   Stacks: {n_stacks}")
        logger.info(f"   Pooling sizes: {pooling_sizes}")
        
        # Créer les stacks
        self.stacks = nn.ModuleList()
        
        for i in range(n_stacks):
            # Ajuster input_size pour être divisible par pooling_size
            pooling = pooling_sizes[i] if i < len(pooling_sizes) else 1
            adjusted_input = (input_size // pooling) * pooling
            
            stack = NHiTSStack(
                input_size=adjusted_input,
                output_size=output_size,
                n_blocks=n_blocks[i],
                hidden_units=mlp_units[i] if i < len(mlp_units) else [512, 512],
                pooling_size=pooling,
                n_theta=n_theta,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm
            )
            self.stacks.append(stack)
        
        # Stocke les tailles ajustées
        self.adjusted_sizes = [
            (input_size // p) * p for p in pooling_sizes
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) ou (batch, input_size)
        
        Returns:
            forecast: (batch, output_size)
        """
        # Aplatir si nécessaire
        if x.dim() == 3:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        # S'assurer que la taille correspond
        if x.shape[1] != self.input_size:
            # Padding ou truncation
            if x.shape[1] < self.input_size:
                x = F.pad(x, (0, self.input_size - x.shape[1]))
            else:
                x = x[:, :self.input_size]
        
        # Passage par les stacks
        residual = x
        forecast_total = torch.zeros(x.shape[0], self.output_size, device=x.device)
        
        for i, stack in enumerate(self.stacks):
            # Ajuster la taille du résidu pour le stack
            adj_size = self.adjusted_sizes[i]
            x_adj = residual[:, :adj_size]
            
            backcast, forecast = stack(x_adj)
            
            # Mettre à jour le résidu (en gardant les dimensions originales)
            residual = residual.clone()
            residual[:, :adj_size] = residual[:, :adj_size] - backcast
            
            forecast_total = forecast_total + forecast
        
        return forecast_total


class NHiTSForecaster(nn.Module):
    """
    Wrapper N-HiTS pour interface avec données 3D (seq_len, n_features).
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int,
        config: Optional[dict] = None
    ):
        super().__init__()
        
        input_size = seq_len * n_features
        
        # Paramètres par défaut
        default_config = {
            'n_blocks': [3, 3, 3],
            'mlp_units': [[512, 512], [512, 512], [512, 512]],
            'pooling_sizes': [4, 2, 1],
            'dropout': 0.1,
            'activation': 'relu',
            'batch_normalization': True
        }
        
        if config:
            default_config.update(config)
        
        self.model = NHiTS(
            input_size=input_size,
            output_size=pred_len,
            n_blocks=default_config['n_blocks'],
            mlp_units=default_config['mlp_units'],
            pooling_sizes=default_config['pooling_sizes'],
            dropout=default_config['dropout'],
            activation=default_config['activation'],
            batch_norm=default_config.get('batch_normalization', True)
        )
        
        self.seq_len = seq_len
        self.n_features = n_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, pred_len)
        """
        return self.model(x)


def create_nhits_from_config(
    n_features: int,
    seq_len: int,
    pred_len: int,
    config_path: str = "configs/model_params.yaml"
) -> NHiTSForecaster:
    """
    Crée un modèle N-HiTS depuis la configuration YAML.
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    params = config.get('n_hits', {})
    
    return NHiTSForecaster(
        n_features=n_features,
        seq_len=seq_len,
        pred_len=pred_len,
        config=params
    )


# Test du module
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    batch_size = 32
    seq_len = 96
    n_features = 12
    pred_len = 96
    
    model = NHiTSForecaster(
        n_features=n_features,
        seq_len=seq_len,
        pred_len=pred_len
    )
    
    x = torch.randn(batch_size, seq_len, n_features)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
