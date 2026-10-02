"""
Modèles de Baseline pour Prévision PV
=====================================
LSTM, GRU, Transformer (avec Positional Encoding), Persistence

Ces modèles servent de références pour comparer avec PatchTST et N-HiTS.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# POSITIONAL ENCODING (pour Transformer)
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoïdal standard.
    
    CRITIQUE: Sans positional encoding, le Transformer ignore
    l'ordre temporel des données!
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# LSTM MULTI-HORIZON
# =============================================================================

class LSTMMultiHorizon(nn.Module):
    """
    LSTM pour prévision multi-horizon.
    
    Architecture:
    - LSTM multi-couches avec dropout
    - Couche de sortie: Linear → pred_len
    """
    
    def __init__(
        self,
        n_features: int,
        pred_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len)
        )
        
        logger.info(f"🔧 LSTM initialisé: hidden={hidden_size}, layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, pred_len)
        """
        out, (h_n, c_n) = self.lstm(x)
        # Utiliser le dernier état
        last_output = out[:, -1, :]
        return self.fc(last_output)


# =============================================================================
# GRU MULTI-HORIZON
# =============================================================================

class GRUMultiHorizon(nn.Module):
    """
    GRU pour prévision multi-horizon.
    """
    
    def __init__(
        self,
        n_features: int,
        pred_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len)
        )
        
        logger.info(f"🔧 GRU initialisé: hidden={hidden_size}, layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        last_output = out[:, -1, :]
        return self.fc(last_output)


# =============================================================================
# TRANSFORMER AMÉLIORÉ (avec Positional Encoding)
# =============================================================================

class TransformerMultiHorizon(nn.Module):
    """
    Transformer pour prévision multi-horizon.
    
    IMPORTANT: Inclut Positional Encoding pour préserver l'ordre temporel.
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pred_len = pred_len
        
        # Projection d'entrée
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional Encoding (CRITIQUE!)
        if positional_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, seq_len + 100, dropout)
        else:
            # Learnable
            self.pos_encoding = nn.Sequential(
                nn.Embedding(seq_len + 100, d_model),
                nn.Dropout(dropout)
            )
            self._learnable_pe = True
        
        self._learnable_pe = positional_encoding != "sinusoidal"
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Tête de sortie
        self.fc = nn.Sequential(
            nn.Linear(d_model * seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len)
        )
        
        self.seq_len = seq_len
        
        logger.info(f"🔧 Transformer initialisé: d_model={d_model}, heads={n_heads}")
        logger.info(f"   ✅ Positional Encoding: {positional_encoding}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, pred_len)
        """
        batch_size = x.shape[0]
        
        # Projection
        x = self.input_projection(x)
        
        # Positional Encoding
        if self._learnable_pe:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.pos_encoding[0](positions)
            x = self.pos_encoding[1](x)
        else:
            x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Flatten et projection finale
        x = x.reshape(batch_size, -1)
        out = self.fc(x)
        
        return out


# =============================================================================
# MODÈLES DE PERSISTENCE (BASELINES)
# =============================================================================

class PersistenceModel:
    """
    Modèle de Persistence (baseline naïf).

    Prédit le dernier kt connu pour tous les horizons futurs.
    Correctement câblé pour extraire kt depuis la matrice de features
    scalée (inverse-transform via feature_scaler + kt_feature_idx).

    Référence : Lorenz et al. (2011), Pedro & Coimbra (2012).
    """

    def __init__(self, pred_len: int):
        self.pred_len = pred_len

    def predict(
        self,
        X: np.ndarray,
        y_last: np.ndarray = None,
        feature_scaler=None,
        kt_feature_idx: int = None,
    ) -> np.ndarray:
        """
        Args:
            X             : (n_samples, seq_len, n_features) — séquences scalées
            y_last        : Derniers kt bruts (n_samples,) — utilisé si fourni
            feature_scaler: sklearn scaler (fit sur X_train) pour l'inverse-transform
            kt_feature_idx: Colonne de kt dans la dimension features de X

        Returns:
            (n_samples, pred_len) — dernier kt répété sur l'horizon
        """
        if y_last is not None:
            # Chemin explicite (e.g. passé depuis un pipeline externe)
            last_values = np.asarray(y_last).flatten()

        elif kt_feature_idx is not None and feature_scaler is not None:
            # Chemin nominal : inverse-transform du dernier pas de temps
            last_step_scaled = X[:, -1, :]                            # (n, n_feat)
            last_step_raw    = feature_scaler.inverse_transform(last_step_scaled)
            last_values      = np.clip(last_step_raw[:, kt_feature_idx], 0.0, None)

        elif X.ndim == 3:
            # Fallback legacy (avant l'inclusion de kt en features) — NE PAS UTILISER
            # en production ; conservé uniquement pour la compatibilité arrière.
            logger.warning(
                "PersistenceModel: kt_feature_idx non fourni — "
                "utilisation de X[:,-1,0] (peut être Wind_Speed, pas kt)."
            )
            last_values = X[:, -1, 0]

        else:
            last_values = X[:, -1]

        return np.tile(last_values.reshape(-1, 1), (1, self.pred_len))


class SmartPersistenceModel:
    """
    Day-ahead ("smart", seasonal) persistence of the clear-sky index.

    Forecasts k_t(t+h) with the value observed at the same time slot one
    seasonal period earlier: k_t(t+h - P * ceil(h / P)), P = steps per day
    (96 at 15 min, 24 at 1 h). The required history is already inside the
    lookback window whenever L >= P, so no external data is needed.

    Référence : Diagne et al. (2013) Solar Energy ; Yang et al. (2020) Solar Energy.
    """

    def __init__(self, pred_len: int, seasonal_period: int = 96):
        self.pred_len = pred_len
        self.seasonal_period = seasonal_period

    def predict(
        self,
        X: np.ndarray,
        feature_scaler=None,
        kt_feature_idx: int = None,
    ) -> np.ndarray:
        """
        Args:
            X              : (n_samples, seq_len, n_features) scaled inputs
            feature_scaler : sklearn scaler fitted on X_train (for inverse-transform)
            kt_feature_idx : column of k_t in the feature dimension of X

        Returns:
            (n_samples, pred_len)
        """
        if kt_feature_idx is None or feature_scaler is None:
            raise ValueError(
                "SmartPersistenceModel needs feature_scaler and kt_feature_idx: "
                "without them it would silently degrade to last-value persistence."
            )
        n, L, n_feat = X.shape
        P = self.seasonal_period
        if P > L:
            raise ValueError(f"seasonal_period ({P}) exceeds the lookback window ({L}).")

        kt = feature_scaler.inverse_transform(X.reshape(-1, n_feat)).reshape(n, L, n_feat)
        kt = np.clip(kt[:, :, kt_feature_idx], 0.0, None)

        # 0-based step h0 forecasts t+1+h0; its reference is P*ceil((h0+1)/P) steps back,
        # i.e. window slot L-1 - (P*ceil((h0+1)/P) - 1 - h0).
        h0 = np.arange(self.pred_len)
        slots = L - 1 - (P * np.ceil((h0 + 1) / P).astype(int) - 1 - h0)
        return kt[:, slots]


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def create_model_from_config(
    model_type: str,
    n_features: int,
    seq_len: int,
    pred_len: int,
    config_path: str = "configs/model_params.yaml"
) -> nn.Module:
    """
    Crée un modèle depuis la configuration YAML.
    
    Args:
        model_type: "lstm", "gru", "transformer"
        n_features: Nombre de features
        seq_len: Longueur de séquence
        pred_len: Horizon de prédiction
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type.lower() == "lstm":
        params = config.get('lstm', {})
        return LSTMMultiHorizon(
            n_features=n_features,
            pred_len=pred_len,
            hidden_size=params.get('hidden_size', 128),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            bidirectional=params.get('bidirectional', False)
        )
    
    elif model_type.lower() == "gru":
        params = config.get('gru', {})
        return GRUMultiHorizon(
            n_features=n_features,
            pred_len=pred_len,
            hidden_size=params.get('hidden_size', 128),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            bidirectional=params.get('bidirectional', False)
        )
    
    elif model_type.lower() == "transformer":
        params = config.get('transformer', {})
        return TransformerMultiHorizon(
            n_features=n_features,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=params.get('d_model', 64),
            n_heads=params.get('n_heads', 4),
            n_layers=params.get('n_layers', 2),
            d_ff=params.get('d_ff', 128),
            dropout=params.get('dropout', 0.1),
            positional_encoding=params.get('positional_encoding', 'sinusoidal')
        )
    
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")


# Alias pour compatibilité arrière
LSTMModel = LSTMMultiHorizon
GRUModel = GRUMultiHorizon
TransformerModel = TransformerMultiHorizon
