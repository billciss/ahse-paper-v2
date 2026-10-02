"""
DataLoaders PyTorch pour Prévision PV
=====================================
Fournit des DataLoaders optimisés pour l'entraînement multi-horizon.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    Dataset PyTorch pour séries temporelles multi-horizon.
    
    Supporte:
    - Entrées 3D (samples, sequence_length, features)
    - Sorties 2D (samples, horizon) pour multi-output
    - Sorties 1D (samples,) pour single-output
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            X: Features shape (n_samples, seq_len, n_features)
            y: Targets shape (n_samples, horizon) ou (n_samples,)
            device: Device PyTorch (None = lazy loading)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.device = device
        
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class FlatTimeSeriesDataset(Dataset):
    """
    Dataset pour modèles non-séquentiels (LightGBM, XGBoost).
    
    Aplatit les séquences: (samples, seq_len, features) → (samples, seq_len * features)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Aplatir X
        n_samples = X.shape[0]
        self.X = X.reshape(n_samples, -1)
        self.y = y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]
    
    def get_flat_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les arrays aplatis pour sklearn."""
        return self.X, self.y


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour l'entraînement.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        batch_size: Taille des mini-batches
        num_workers: Nombre de workers pour le chargement
        pin_memory: Utiliser la mémoire épinglée pour GPU
        device: Device PyTorch
        seed: Random seed for reproducible shuffling
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    # Create generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device is not None and device.type == 'cuda',
        generator=g  # For reproducibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device is not None and device.type == 'cuda'
    )
    
    logger.info(f"✅ DataLoaders créés:")
    logger.info(f"   Train: {len(train_loader)} batches")
    logger.info(f"   Val: {len(val_loader)} batches")
    
    return train_loader, val_loader


def prepare_flat_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Prépare les données aplaties pour les modèles tree-based.
    
    Returns:
        Dict avec X_train_flat, y_train, X_val_flat, y_val, X_test_flat
    """
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    
    return {
        'X_train_flat': X_train.reshape(n_train, -1),
        'y_train': y_train,
        'X_val_flat': X_val.reshape(n_val, -1),
        'y_val': y_val,
        'X_test_flat': X_test.reshape(n_test, -1)
    }


class InfiniteDataLoader:
    """
    DataLoader qui boucle infiniment pour l'entraînement.
    """
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
    
    def __iter__(self):
        return self
