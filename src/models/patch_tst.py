"""
PatchTST - Patch Time Series Transformer
=========================================
Implémentation du modèle PatchTST avec Channel Independence.

Référence: Nie et al., "A Time Series is Worth 64 Words: 
           Long-term Forecasting with Transformers" (ICLR 2023)

Caractéristiques clés:
1. Patching: Segmente la série en patchs pour réduire la complexité
2. Channel Independence: Traite chaque feature séparément
3. Positional Encoding: Préserve l'information temporelle
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoïdal classique.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Créer la matrice de positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Positional Encoding appris (learnable)."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Transforme la série temporelle en séquence de patchs.
    
    Input: (batch, seq_len, n_features)
    Output: (batch, n_patches, d_model)
    """
    
    def __init__(
        self,
        n_features: int,
        patch_len: int,
        stride: int,
        d_model: int,
        padding_patch: str = 'end'
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        # Projection linéaire: patch → d_model
        self.value_embedding = nn.Linear(patch_len * n_features, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: (batch, seq_len, n_features)
        
        Returns:
            patches: (batch, n_patches, d_model)
            n_patches: Nombre de patchs
        """
        batch_size, seq_len, n_features = x.shape
        
        # Padding si nécessaire
        if self.padding_patch == 'end':
            pad_len = (self.stride - (seq_len - self.patch_len) % self.stride) % self.stride
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
        
        # Créer les patchs avec unfold
        # (batch, seq_len, n_features) → (batch, n_patches, patch_len, n_features)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # x: (batch, n_patches, n_features, patch_len)
        
        n_patches = x.shape[1]
        
        # Flatten et projection
        # (batch, n_patches, n_features * patch_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, n_patches, -1)
        
        # Projection vers d_model
        x = self.value_embedding(x)
        
        return x, n_patches


class PatchTSTEncoder(nn.Module):
    """
    Encodeur Transformer pour PatchTST.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, d_model)
        Returns:
            (batch, n_patches, d_model)
        """
        x = self.encoder(x)
        x = self.norm(x)
        return x


class PatchTST(nn.Module):
    """
    PatchTST - State-of-the-Art Transformer pour séries temporelles.
    
    Architecture:
    1. Patch Embedding: Segmente l'entrée en patchs
    2. Positional Encoding: Ajoute l'information de position
    3. Transformer Encoder: Capture les dépendances
    4. Flatten + Linear: Projection vers l'horizon de prédiction
    
    Args:
        n_features: Nombre de features d'entrée
        seq_len: Longueur de la séquence d'entrée
        pred_len: Horizon de prédiction (H)
        patch_len: Taille des patchs
        stride: Pas entre les patchs
        d_model: Dimension du modèle
        n_heads: Nombre de têtes d'attention
        n_layers: Nombre de couches Transformer
        d_ff: Dimension feedforward
        dropout: Taux de dropout
        channel_independence: Si True, traite chaque channel séparément
        positional_encoding: "sinusoidal" ou "learnable"
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        channel_independence: bool = True,
        positional_encoding: str = "learnable"
    ):
        super().__init__()
        
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channel_independence = channel_independence
        
        # Nombre de patchs
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        logger.info(f"🔧 PatchTST initialisé:")
        logger.info(f"   Input: ({seq_len}, {n_features}) → Output: ({pred_len},)")
        logger.info(f"   Patchs: {self.n_patches} (len={patch_len}, stride={stride})")
        logger.info(f"   Channel Independence: {channel_independence}")
        
        if channel_independence:
            # Mode Channel Independence: 1 feature par patch
            self.patch_embedding = nn.Linear(patch_len, d_model)
        else:
            # Mode standard: tous les features par patch
            self.patch_embedding = PatchEmbedding(
                n_features=n_features,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model
            )
        
        # Positional Encoding
        max_len = max(self.n_patches * n_features + 10, 1000)
        if positional_encoding == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer Encoder
        self.encoder = PatchTSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Head de prédiction
        if channel_independence:
            # Une sortie par feature, puis agrégation
            self.head = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.n_patches * d_model * n_features, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, pred_len)
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.n_patches * d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, pred_len)
            )
        
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        
        Returns:
            predictions: (batch, pred_len)
        """
        batch_size = x.shape[0]
        
        if self.channel_independence:
            # Channel Independence: traiter chaque feature séparément
            # x: (batch, seq_len, n_features)
            
            # Créer les patchs pour chaque channel
            # (batch, n_features, seq_len)
            x = x.permute(0, 2, 1)
            
            # Unfold pour créer les patchs
            # (batch, n_features, n_patches, patch_len)
            x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
            
            # Reshape pour le Transformer
            # (batch, n_features * n_patches, patch_len)
            n_feat = x.shape[1]
            n_patch = x.shape[2]
            x = x.reshape(batch_size, n_feat * n_patch, self.patch_len)
            
            # Projection des patchs
            x = self.patch_embedding(x)  # (batch, n_feat * n_patch, d_model)
            
        else:
            # Mode standard
            x, _ = self.patch_embedding(x)
        
        # Positional Encoding
        x = self.pos_encoding(x)
        
        # Transformer Encoder
        x = self.encoder(x)
        
        # Head de prédiction
        out = self.head(x)
        
        return out


class PatchTSTForecaster(nn.Module):
    """
    Wrapper pour PatchTST avec interface simplifiée.
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int,
        config: Optional[dict] = None
    ):
        super().__init__()
        
        # Paramètres par défaut
        default_config = {
            'patch_len': 16,
            'stride': 8,
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'channel_independence': True,
            'positional_encoding': 'learnable'
        }
        
        if config:
            default_config.update(config)
        
        self.model = PatchTST(
            n_features=n_features,
            seq_len=seq_len,
            pred_len=pred_len,
            **default_config
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_patchtst_from_config(
    n_features: int,
    seq_len: int,
    pred_len: int,
    config_path: str = "configs/model_params.yaml"
) -> PatchTST:
    """
    Crée un modèle PatchTST depuis la configuration YAML.
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    params = config.get('patch_tst', {})
    
    return PatchTST(
        n_features=n_features,
        seq_len=seq_len,
        pred_len=pred_len,
        patch_len=params.get('patch_len', 16),
        stride=params.get('stride', 8),
        d_model=params.get('d_model', 128),
        n_heads=params.get('n_heads', 8),
        n_layers=params.get('n_layers', 3),
        d_ff=params.get('d_ff', 256),
        dropout=params.get('dropout', 0.2),
        channel_independence=params.get('channel_independence', True),
        positional_encoding=params.get('positional_encoding', 'learnable')
    )
