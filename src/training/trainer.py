"""
Trainer Multi-Horizon pour Prévision PV
=======================================
Boucle d'entraînement optimisée avec:
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Support multi-output
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, List, Callable
import logging
import time
import yaml

logger = logging.getLogger(__name__)


class MultiHorizonTrainer:
    """
    Entraîneur pour modèles de prévision multi-horizon (Deep Learning).
    
    Features:
    - Early stopping intelligent
    - Scheduling du learning rate
    - Logging détaillé
    - Support pour différentes loss functions
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger config si fournie
        if config_path:
            self._load_config(config_path)
        else:
            self.epochs = 100
            self.batch_size = 64
            self.patience = 15
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.gradient_clip = 1.0
        
        logger.info(f"🖥️ Trainer initialisé sur {self.device}")
    
    def _load_config(self, config_path: str):
        """Charge la configuration depuis YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        training = config.get('training_global', {})
        self.epochs = training.get('epochs', 100)
        self.batch_size = training.get('batch_size', 64)
        # CORRECTION: Suppression du 's' parasite qui causait la SyntaxError
        self.patience = training.get('early_stopping_patience', 15)
        self.learning_rate = training.get('learning_rate_default', 0.001)
        self.weight_decay = training.get('weight_decay', 1e-5)
        self.gradient_clip = training.get('gradient_clip_max_norm', 1.0)

    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        patience: Optional[int] = None,
        loss_fn: Optional[nn.Module] = None,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict]:
        """Entraîne un modèle PyTorch avec early stopping."""
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate
        patience = patience or self.patience
        loss_fn = loss_fn or nn.MSELoss()
        
        model = model.to(self.device)
        
        # 1. Convertir en tenseurs (UNIQUEMENT LE TRAIN)
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        
        # 2. DataLoader (Train seulement) - with reproducibility
        train_dataset = TensorDataset(X_train_t, y_train_t)
        
        # Create generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(42)  # Fixed seed for reproducibility
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            generator=g,  # Reproducible shuffling
            drop_last=True
        )
        
        # Optimizer et scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Training loop state
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                if outputs.dim() != y_batch.dim():
                    outputs = outputs.squeeze() if outputs.dim() > y_batch.dim() else outputs
                
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            # Guard: if dataset smaller than batch_size and drop_last=True,
            # n_batches can be 0 (e.g. when called with X_train[:1] to rebuild
            # model architecture for checkpoint loading).
            if n_batches == 0:
                break   # Nothing trained this epoch — exit training loop early
            train_loss /= n_batches
            
            # === LA BOUCLE DE VALIDATION SÉCURISÉE (BATCHÉE) ===
            model.eval()
            val_loss = 0
            with torch.no_grad():
                val_batch_size = 32 
                for i in range(0, len(X_val), val_batch_size):
                    # On convertit et monte de petits morceaux sur le GPU
                    X_v = torch.FloatTensor(X_val[i:i+val_batch_size]).to(self.device)
                    y_v = torch.FloatTensor(y_val[i:i+val_batch_size]).to(self.device)
                    
                    outputs = model(X_v)
                    
                    if outputs.dim() != y_v.dim():
                        outputs = outputs.squeeze() if outputs.dim() > y_v.dim() else outputs
                        
                    val_loss += loss_fn(outputs, y_v).item() * len(X_v)
                
                val_loss /= len(X_val)
            
            # Historique
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Scheduler
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        logger.info(f"   ⏹️ Early stopping epoch {epoch + 1}")
                    break
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(f"   Epoch {epoch + 1:3d}/{epochs} | "
                           f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
        
        # Restaurer le meilleur modèle
        if best_state:
            model.load_state_dict(best_state)
            model = model.to(self.device)
        
        elapsed = time.time() - start_time
        if verbose:
            logger.info(f"   ✅ Entraînement terminé en {elapsed:.1f}s")
            logger.info(f"   📊 Best val loss: {best_val_loss:.4f}")
        
        return model, history
    
    def predict(
        self,
        model: nn.Module,
        X: np.ndarray,
        batch_size: int = 256
    ) -> np.ndarray:
        model.eval()
        model = model.to(self.device)
        n_samples = X.shape[0]
        predictions = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                X_batch = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
                preds = model(X_batch).cpu().numpy()
                predictions.append(preds)
        return np.concatenate(predictions, axis=0)
    
    def train_multiple_models(
        self,
        models: Dict[str, nn.Module],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs
    ) -> Dict[str, Tuple[nn.Module, Dict]]:
        results = {}
        for name, model in models.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"🧠 ENTRAÎNEMENT: {name}")
            logger.info("=" * 60)
            trained_model, history = self.train(
                model, X_train, y_train, X_val, y_val, **kwargs
            )
            results[name] = (trained_model, history)
        return results


class TreeModelTrainer:
    """
    Entraîneur pour modèles tree-based (LightGBM, XGBoost).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self._load_config(config_path)
        else:
            self.lightgbm_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 8}
            self.xgboost_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6}
    
    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        gb_config = config.get('gradient_boosting', {})
        self.lightgbm_params = gb_config.get('lightgbm', {})
        self.xgboost_params = gb_config.get('xgboost', {})

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = 'lightgbm',
        params: Optional[Dict] = None,
        seed: int = 42
    ):
        """Méthode dispatcher (Indentation CORRIGÉE pour appartenir à la classe)"""
        if model_type.lower() == 'lightgbm':
            return self.train_lightgbm(X_train, y_train, X_val, y_val, params, seed=seed)
        elif model_type.lower() == 'xgboost':
            return self.train_xgboost(X_train, y_train, X_val, y_val, params, seed=seed)
        else:
            raise ValueError(f"Modèle d'arbre non supporté : {model_type}")

    def train_lightgbm(self, X_train, y_train, X_val, y_val, params=None, seed=42):
        try:
            import lightgbm as lgb
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            logger.error("❌ LightGBM non installé")
            return None
        
        params = params or self.lightgbm_params.copy()
        params.pop('multi_output', None)
        
        # Add reproducibility parameters
        params['seed'] = seed
        params['bagging_seed'] = seed
        params['feature_fraction_seed'] = seed
        params['data_random_seed'] = seed
        # Strategy: let LightGBM use ALL cores internally (n_jobs=-1) for each
        # of the 96 sequential models. Much more memory-efficient than spawning
        # 16 worker processes (each duplicating the 30K×1344 feature matrix).
        # Remove force_row_wise so LightGBM can use histogram parallelism.
        # Seed ensures reproducibility without needing deterministic=True.
        params.pop('deterministic', None)
        params.pop('force_row_wise', None)
        params['n_jobs'] = -1   # LightGBM internal threading — all cores, no forking

        logger.info("🌲 Entraînement LightGBM Multi-Output (internal n_jobs=-1, sequential 96 outputs)...")

        # Aplatissage
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if X_train.ndim == 3 else X_train
        X_val_flat = X_val.reshape(X_val.shape[0], -1) if X_val.ndim == 3 else X_val

        base_model = lgb.LGBMRegressor(**params, verbose=-1)
        if y_train.ndim == 1 or y_train.shape[1] == 1:
            model = base_model
            model.fit(X_train_flat, y_train.ravel(), eval_set=[(X_val_flat, y_val.ravel())],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            model = MultiOutputRegressor(base_model, n_jobs=1)  # sequential, each model multi-threaded
            model.fit(X_train_flat, y_train)
        
        logger.info("   ✅ LightGBM entraîné")
        return model

    def train_xgboost(self, X_train, y_train, X_val, y_val, params=None, seed=42):
        try:
            import xgboost as xgb
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            logger.error("❌ XGBoost non installé")
            return None
        
        params = params or self.xgboost_params.copy()
        params.pop('multi_output', None)
        
        # Add reproducibility parameters
        params['seed'] = seed
        params['random_state'] = seed
        
        # Same strategy as LightGBM: internal threading, sequential outer loop
        params['n_jobs'] = -1   # XGBoost internal threading — all cores
        logger.info("🌲 Entraînement XGBoost Multi-Output (internal n_jobs=-1, sequential 96 outputs)...")

        X_train_flat = X_train.reshape(X_train.shape[0], -1) if X_train.ndim == 3 else X_train
        X_val_flat = X_val.reshape(X_val.shape[0], -1) if X_val.ndim == 3 else X_val

        base_model = xgb.XGBRegressor(**params, verbosity=0)
        if y_train.ndim == 1 or y_train.shape[1] == 1:
            model = base_model
            model.fit(X_train_flat, y_train.ravel(), eval_set=[(X_val_flat, y_val.ravel())], verbose=False)
        else:
            model = MultiOutputRegressor(base_model, n_jobs=1)  # sequential, each model multi-threaded
            model.fit(X_train_flat, y_train)
        
        logger.info("   ✅ XGBoost entraîné")
        return model

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
        return model.predict(X_flat)


# Alias pour compatibilité
ModelTrainer = MultiHorizonTrainer