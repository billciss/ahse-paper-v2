#!/usr/bin/env python
"""
AHSE v2 - training and evaluation pipeline
==========================================
Trains the base models, builds the AHSE per-horizon-block selection on the validation period,
evaluates every model on the test period and writes tables and a prediction cache.

Usage:
------
    # Yulara (four uncurtailed sites), hourly, without and with GFS covariates
    python main.py --dataset yulara_neighbours --horizon 1h --models all --seed 42
    python main.py --dataset yulara_neighbours --horizon 1h --models all --seed 42 \
        --gfs data_external/gfs/gfs_yulara_DSWRF.csv data_external/gfs/gfs_yulara_TCDC.csv \
        --results-dir results/yulara_neighbours_gfs

    # NIST Ground array (PVDAQ 4902)
    python main.py --dataset nist_pvdaq --horizon 1h --models all --results-dir results/nist

Data: data/processed/<dataset>_<horizon>.csv (see scripts/prepare_yulara_sites.py and
scripts/prepare_nist.py); configuration: configs/data_config_<site>.yaml.
Results are saved to <results-dir>/{models,tables,figures}/ (default results/<dataset>).
"""
import joblib
import argparse
import os
import sys
import json
import yaml
import logging
import gc      # Pour libérer la RAM (CPU)
import torch   # Pour libérer la VRAM (GPU)
from scipy import stats
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pvlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# REPRODUCIBILITY - Import from centralized module
# =============================================================================
from src.utils.reproducibility import init_reproducibility, set_global_seed, SEED_CONFIG

GLOBAL_SEED = SEED_CONFIG.get('global_seed', 42)

from src.data_engine import (
    ClearSkyCalculator,
    MultiHorizonPreprocessor,
    load_and_prepare_multi_horizon
)
from src.models import (
    create_patchtst_from_config,
    create_nhits_from_config,
    LSTMMultiHorizon,
    GRUMultiHorizon,
    TransformerMultiHorizon,
    PersistenceModel,
    SmartPersistenceModel,
    AHSEMultiHorizon
)
from src.models.honest_selection import HonestBlockSelector, default_blocks
from src.data_engine.nwp_features import gfs_target_features, load_gfs
from src.training import MultiHorizonTrainer, TreeModelTrainer
from src.evaluation import MetricsFactory, HorizonErrorAnalyzer
from src.visualization import (
    plot_forecast_comparison,
    plot_horizon_degradation,
    plot_model_comparison_heatmap,
    plot_training_history,
    save_figure_for_publication
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PVForecastingPipeline:
    """
    Pipeline complet d'entraînement et d'évaluation SOTA.
    """
    
    AVAILABLE_MODELS = [
        'PatchTST', 'N-HiTS', 
        'LSTM', 'GRU', 'Transformer',
        'LightGBM', 'XGBoost',
        'Persistence', 'SmartPersistence',
        'AHSE'
    ]
    
    def __init__(
        self,
        data_config_path: str = 'configs/data_config_yulara_neighbours.yaml',
        model_config_path: str = 'configs/model_params.yaml',
        results_dir: str = 'results',
        data_file: Optional[str] = None,
        end: Optional[str] = None,
        gfs_files: Optional[Tuple[str, str]] = None,
        seed: int = 42
    ):
        self.data_config_path = data_config_path
        self.data_file = data_file      # overrides <processed_dir>/<results_dir name>_<horizon>.csv
        self.end = end                  # last date kept (e.g. end of the NWP archive)
        self.gfs_files = gfs_files      # (DSWRF csv, TCDC csv): GFS covariates for tree models
        self.nwp = None                 # split -> (n, H*4) GFS features, set in load_data
        self.seed = seed                # also seeds the tree models (they used a fixed 42 before)
        self.model_config_path = model_config_path
        self.results_dir = Path(results_dir)
        
        # Create output directories
        self.models_dir = self.results_dir / 'models'
        self.figures_dir = self.results_dir / 'figures'
        self.tables_dir = self.results_dir / 'tables'
        
        for d in [self.models_dir, self.figures_dir, self.tables_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load configs
        self.data_config = self._load_config(data_config_path)
        self.model_config = self._load_config(model_config_path)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Device: {self.device}")
        
        # Results storage
        self.results = {}
        self.predictions = {}       # prédictions sur test set
        self.predictions_val = {}   # prédictions sur val set (pour sélection AHSE)
        self.trained_models = {}
    
    def _load_config(self, path: str) -> dict:
        """Load YAML configuration."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, horizon: str = '15min') -> Dict:
        """
        Load and prepare data with Clear Sky Index normalization.
        """
        logger.info("=" * 60)
        logger.info("📊 CHARGEMENT DES DONNÉES")
        logger.info("=" * 60)
        
        # 1. Récupérer le dossier des données depuis la config
        data_dir = self.data_config.get('data', {}).get('processed_dir', 'data/processed')
        
        # 2. Fallback dataset name: the results folder name (main() passes the data file
        #    data/processed/<dataset>_<horizon>.csv explicitly)
        dataset_name = self.results_dir.name
        
        # 3. Data file (e.g. data/processed/yulara_neighbours_1h.csv)
        data_file = self.data_file or os.path.join(data_dir, f"{dataset_name}_{horizon}.csv")
        
        logger.info(f"🔎 Recherche du fichier : {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"❌ Fichier non trouvé: {data_file}\n"
                f"   → Fichiers disponibles dans {data_dir}: {os.listdir(data_dir)}"
            )
        
        # Charger avec le préprocesseur multi-horizon
        self.data = load_and_prepare_multi_horizon(
            data_file,
            config_path=self.data_config_path,
            horizon=horizon,
            end=self.end
        )

        if self.gfs_files is not None:
            # Day-ahead GFS covariates per target step (latest run published at the issue
            # time, see src/data_engine/nwp_features.py). Used by the tree models; their
            # clearness index is also a stand-alone ensemble member ('GFS').
            loc_cfg = self.data_config['location']
            loc = pvlib.location.Location(loc_cfg['latitude'], loc_cfg['longitude'],
                                          tz=loc_cfg['timezone'], altitude=loc_cfg.get('altitude', 0))
            gfs = load_gfs(*self.gfs_files)
            H = self.data['y_train'].shape[1]
            self.nwp, self.nwp_kc = {}, {}
            for split in ['train', 'val', 'test']:
                f = gfs_target_features(gfs, self.data['time_index'], self.data['target_start_rows'][split],
                                        H, pd.Timedelta(horizon), loc)
                self.nwp[split] = np.hstack([f['dswrf'], f['tcdc'], f['cs'], f['kc']])
                self.nwp_kc[split] = f['kc']
                logger.info(f"   GFS {split}: {np.isnan(f['dswrf']).mean():.1%} of targets without a run value")
        
        logger.info(f"✅ Données chargées avec succès ({dataset_name} - {horizon})")
        return self.data
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """
        Train a single model.
        """
        logger.info(f"\n🔧 Training {model_name}...")
        
        n_features = X_train.shape[2]
        seq_len = X_train.shape[1]
        pred_len = y_train.shape[1] if y_train.ndim > 1 else 1
        
        result = {'model': None, 'history': None, 'time': 0}
        start_time = datetime.now()
        
        try:
            if model_name == 'PatchTST':
                model = create_patchtst_from_config(
                    n_features=n_features,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    config_path=self.model_config_path
                )
                trainer = MultiHorizonTrainer(
                    device=self.device,
                    config_path=self.model_config_path
                )
                model, history = trainer.train(
                    model, X_train, y_train, X_val, y_val
                )
                result['model'] = model
                result['history'] = history
                
            elif model_name == 'N-HiTS':
                model = create_nhits_from_config(
                    n_features=n_features,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    config_path=self.model_config_path
                )
                trainer = MultiHorizonTrainer(
                    device=self.device,
                    config_path=self.model_config_path
                )
                model, history = trainer.train(
                    model, X_train, y_train, X_val, y_val
                )
                result['model'] = model
                result['history'] = history
                
            elif model_name == 'LSTM':
                # On crée une copie pour ne pas modifier la config globale
                config = self.model_config.get('lstm', {}).copy()
                # On retire 'learning_rate' pour qu'il ne soit pas envoyé au modèle
                lr = config.pop('learning_rate', 0.001)
                
                model = LSTMMultiHorizon(
                    n_features=n_features,
                    pred_len=pred_len,
                    **config  # Ne contient plus learning_rate
                )
                trainer = MultiHorizonTrainer(
                    device=self.device,
                    config_path=self.model_config_path
                )
                # On passe le lr ici au trainer
                model, history = trainer.train(
                    model, X_train, y_train, X_val, y_val,
                    learning_rate=lr
                )
                result['model'] = model
                result['history'] = history
                    
            elif model_name == 'GRU':
                config = self.model_config.get('gru', {}).copy()
                lr = config.pop('learning_rate', 0.001)
                
                model = GRUMultiHorizon(
                    n_features=n_features,
                    pred_len=pred_len,
                    **config
                )
                trainer = MultiHorizonTrainer(
                    device=self.device,
                    config_path=self.model_config_path
                )
                model, history = trainer.train(
                    model, X_train, y_train, X_val, y_val,
                    learning_rate=lr
                )
                result['model'] = model
                result['history'] = history
                    
            elif model_name == 'Transformer':
                config = self.model_config.get('transformer', {}).copy()
                lr = config.pop('learning_rate', 0.001)
                
                model = TransformerMultiHorizon(
                    n_features=n_features,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    **config
                )
                trainer = MultiHorizonTrainer(
                    device=self.device,
                    config_path=self.model_config_path
                )
                model, history = trainer.train(
                    model, X_train, y_train, X_val, y_val,
                    learning_rate=lr
                )
                result['model'] = model
                result['history'] = history
                
            elif model_name in ['LightGBM', 'XGBoost']:
                trainer = TreeModelTrainer(config_path=self.model_config_path)
                model = trainer.train(
                    self._tree_features(X_train, 'train'),
                    y_train,
                    self._tree_features(X_val, 'val'),
                    y_val,
                    model_type=model_name.lower(),
                    seed=self.seed
                )
                result['model'] = model
                result['history'] = {'train_loss': [], 'val_loss': []}
                
            elif model_name == 'Persistence':
                model = PersistenceModel(pred_len=pred_len)
                result['model'] = model
                result['history'] = {'train_loss': [], 'val_loss': []}
                
            elif model_name == 'SmartPersistence':
                # One day in steps of the current resolution (96 at 15 min, 24 at 1 h).
                seasonal = int(pd.Timedelta('1D') / pd.Timedelta(self.horizon))
                model = SmartPersistenceModel(pred_len=pred_len, seasonal_period=seasonal)
                result['model'] = model
                result['history'] = {'train_loss': [], 'val_loss': []}
            
            result['time'] = (datetime.now() - start_time).total_seconds()
            logger.info(f"   ✅ {model_name} trained in {result['time']:.1f}s")
            
        except Exception as e:
            logger.error(f"   ❌ {model_name} failed: {e}")
            raise
        
        return result
    
    def _tree_features(self, X: np.ndarray, split: str) -> np.ndarray:
        """Flattened lookback window, plus the GFS covariates of the split when enabled."""
        X_flat = X.reshape(X.shape[0], -1)
        if self.nwp is None:
            return X_flat
        if len(self.nwp[split]) != len(X_flat):
            raise ValueError(f"GFS features of split '{split}' do not match X "
                             f"({len(self.nwp[split])} vs {len(X_flat)})")
        return np.hstack([X_flat, self.nwp[split]])

    def _tree_feature_names(self) -> List[str]:
        """Names of the columns built by _tree_features (lookback steps x features, then GFS)."""
        L = self.data['X_train'].shape[1]
        names = [f"{f}[t-{L - 1 - l}]" for l in range(L) for f in self.data['feature_names']]
        if self.nwp is not None:
            H = self.data['y_train'].shape[1]
            names += [f"gfs_{v}[t+{h + 1}]" for v in ['dswrf', 'tcdc', 'cs', 'kc'] for h in range(H)]
        return names

    def predict(self, model, model_name: str, X_test: np.ndarray, split: str = 'test') -> np.ndarray:
        """
        Génère des prédictions. Ajout d'une gestion par lots (batching) pour 
        éviter la saturation de la mémoire GPU sur les gros datasets.
        """
        if model_name in ['LightGBM', 'XGBoost']:
            return model.predict(self._tree_features(X_test, split))
            
        elif model_name in ['Persistence', 'SmartPersistence']:
            # Passer le scaler + l'indice de kt pour que le modèle lise
            # la vraie dernière valeur de kt (et non Wind_Speed ou autre).
            feature_scaler  = self.data.get('feature_scaler')
            kt_feature_idx  = self.data.get('kt_feature_idx')
            return model.predict(
                X_test,
                feature_scaler=feature_scaler,
                kt_feature_idx=kt_feature_idx,
            )
            
        else:
            # --- MODIFICATION POUR ROBUSTESSE GPU (Batch Processing) ---
            model.eval()
            predictions = []
            # On traite par petits paquets de 256 lignes (sécuritaire pour RTX 3050)
            batch_size = 256 
            
            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    # Extraction du petit lot
                    X_batch = X_test[i:i + batch_size]
                    X_tensor = torch.from_numpy(X_batch).float().to(self.device)
                    
                    # Prédiction sur le lot
                    preds = model(X_tensor).cpu().numpy()
                    predictions.append(preds)
                    
                    # Nettoyage régulier de la mémoire cache du GPU
                    if i % 1024 == 0:
                        torch.cuda.empty_cache()
            
            # Reconstruction du tableau final (on empile les paquets)
            return np.vstack(predictions)
    
    def evaluate(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        daytime_mask: np.ndarray = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Évalue tous les modèles pour le Global (24h) ET le One-Step (t+1).
        """
        logger.info("\n📊 ÉVALUATION DOUBLE : GLOBAL & ONE-STEP")
        
        global_results = []
        onestep_results = []
        
        # FSS references: last-value k_t persistence (Skill_Score) and day-ahead
        # seasonal persistence (Skill_Score_Daily), both via forecast_skill_score().
        y_pers = predictions.get('Persistence')
        y_pers_t1 = y_pers[:, 0].reshape(-1, 1) if y_pers is not None else None
        y_smart = predictions.get('SmartPersistence')
        y_smart_t1 = y_smart[:, 0].reshape(-1, 1) if y_smart is not None else None

        for model_name, y_pred in predictions.items():
            # 1. Évaluation GLOBALE
            m_glob = MetricsFactory.compute_all(
                y_true, y_pred, daytime_mask, y_persistence=y_pers,
                y_smart_persistence=y_smart
            )
            m_glob['Model'] = model_name
            global_results.append(m_glob)

            # 2. Évaluation ONE-STEP (t+1)
            y_t1 = y_true[:, 0].reshape(-1, 1)
            p_t1 = y_pred[:, 0].reshape(-1, 1)
            mask_t1 = daytime_mask[:, 0] if daytime_mask is not None else None
            m_t1 = MetricsFactory.compute_all(
                y_t1, p_t1, mask_t1, y_persistence=y_pers_t1,
                y_smart_persistence=y_smart_t1
            )
            m_t1['Model'] = model_name
            onestep_results.append(m_t1)
            
            r2_val = m_glob.get('R2', 0)
            r2_t1_val = m_t1.get('R2', 0)
            logger.info(f"   {model_name}: Global R2={r2_val:.4f} | One-Step R2={r2_t1_val:.4f}")

        df_global = pd.DataFrame(global_results).sort_values('R2', ascending=False)
        df_onestep = pd.DataFrame(onestep_results).sort_values('R2', ascending=False)
        
        return {
            'global': df_global,
            'onestep': df_onestep
        }
    
    def run_full_pipeline(
        self,
        horizon: str = '15min',
        models: List[str] = None,
        skip_training: bool = False
    ) -> Dict:
        """
        Run complete training and evaluation pipeline.
        """
        logger.info("=" * 60)
        logger.info("🚀 PV FORECASTING SOTA PIPELINE")
        logger.info(f"   Horizon: {horizon}")
        logger.info(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)
        
        self.horizon = horizon   # stocké pour _run_statistical_validation

        # Load data
        data = self.load_data(horizon)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        daytime_mask = data.get('daytime_mask')
        
        # Select models
        if models is None or 'all' in models:
            models = ['PatchTST', 'N-HiTS', 'LSTM', 'GRU', 
                     'LightGBM', 'XGBoost', 'Persistence',
                     'SmartPersistence', 'AHSE']
        
        # Train models
        if not skip_training:
            for model_name in models:
                if model_name == 'AHSE':
                    # AHSE is an ensemble — built after all base models are trained.
                    # Skip here; it is constructed below after predictions are generated.
                    continue
                elif model_name in ['Persistence', 'SmartPersistence']:
                    # No training needed
                    self.trained_models[model_name] = self.train_model(
                        model_name, X_train, y_train, X_val, y_val
                    )
                else:
                    # ── Checkpoint resume: skip training if model already saved ──
                    ckpt_joblib = self.models_dir / f"{model_name}_{horizon}.joblib"
                    ckpt_torch  = self.models_dir / f"{model_name}_{horizon}.pt"

                    # LightGBM / XGBoost: load from joblib checkpoint
                    if ckpt_joblib.exists() and model_name in ['LightGBM', 'XGBoost']:
                        logger.info(f"   ⏭️  {model_name}: checkpoint found, skipping training.")
                        loaded_model = joblib.load(ckpt_joblib)
                        self.trained_models[model_name] = {
                            'model': loaded_model, 'history': None, 'time': 0
                        }
                        continue

                    # DL models (PatchTST, N-HiTS, LSTM, GRU, Transformer):
                    # rebuild architecture with 1 dummy sample, then load weights.
                    elif ckpt_torch.exists() and model_name not in ['LightGBM', 'XGBoost',
                                                                     'Persistence', 'SmartPersistence']:
                        logger.info(f"   ⏭️  {model_name}: checkpoint found, skipping training.")
                        result_tmp = self.train_model(model_name, X_train[:1], y_train[:1],
                                                      X_val[:1], y_val[:1])
                        result_tmp['model'].load_state_dict(
                            torch.load(str(ckpt_torch), map_location=self.device,
                                       weights_only=True)
                        )
                        result_tmp['model'].to(self.device)
                        result_tmp['time'] = 0
                        self.trained_models[model_name] = result_tmp
                        continue
                    # ─────────────────────────────────────────────────────────────

                    result = self.train_model(
                        model_name, X_train, y_train, X_val, y_val
                    )
                    self.trained_models[model_name] = result

                    # ── Save checkpoint immediately after training ────────────────
                    if result['model'] is not None:
                        if model_name in ['LightGBM', 'XGBoost']:
                            joblib.dump(result['model'], ckpt_joblib)
                            logger.info(f"   💾 Checkpoint saved: {ckpt_joblib.name}")
                        else:
                            torch.save(result['model'].state_dict(), str(ckpt_torch))
                            logger.info(f"   💾 Checkpoint saved: {ckpt_torch.name}")
                    # ─────────────────────────────────────────────────────────────

                    # Save training history plot
                    if result['history'] and result['history'].get('train_loss'):
                        fig = plot_training_history(
                            result['history'],
                            model_name=model_name
                        )
                        save_figure_for_publication(
                            fig,
                            f"training_{model_name}_{horizon}",
                            str(self.figures_dir)
                        )
                        plt.close(fig)

                # --- NETTOYAGE MÉMOIRE (À l'intérieur de la boucle 'for') ---
                # Ce bloc doit être aligné verticalement avec le 'if model_name' ci-dessus
                logger.info(f"   🧹 Nettoyage de la mémoire après {model_name}...")
                
                if 'result' in locals():
                    del result
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # ------------------------------------------------------------

        # Generate predictions (Une fois que tous les modèles sont entraînés et nettoyés)
        logger.info("\n🔮 GÉNÉRATION DES PRÉDICTIONS")

        # Supprimer les références aux objets lourds
        if 'result' in locals():
            del result

        # Libérer la RAM (Système)
        gc.collect()

        # Libérer la VRAM (Carte Graphique)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Prédictions sur TEST set (pour évaluation finale)
        logger.info("\n🔮 GÉNÉRATION DES PRÉDICTIONS — TEST SET")
        for model_name, result in self.trained_models.items():
            if result['model'] is not None:
                self.predictions[model_name] = self.predict(
                    result['model'], model_name, X_test
                )

        # Prédictions sur VAL set (pour sélection AHSE — jamais vues à l'évaluation)
        logger.info("\n🔮 GÉNÉRATION DES PRÉDICTIONS — VAL SET (sélection AHSE)")
        for model_name, result in self.trained_models.items():
            if result['model'] is not None:
                self.predictions_val[model_name] = self.predict(
                    result['model'], model_name, X_val, split='val'
                )

        # GFS clearness index as a stand-alone member; where the run has no value the
        # operational fallback is day-ahead persistence.
        if self.nwp is not None and 'SmartPersistence' in self.predictions:
            for split, store in [('val', self.predictions_val), ('test', self.predictions)]:
                kc = self.nwp_kc[split]
                store['GFS'] = np.where(np.isnan(kc), store['SmartPersistence'], kc)

        # === BLOC AHSE — PROTOCOLE HONNÊTE VAL/TEST ===
        if 'AHSE' in models or 'all' in models:
            logger.info("\n🧠 CONSTRUCTION DE L'ENSEMBLE ADAPTATIF (AHSE)")

            # 1. Récupérer les paramètres du fichier YAML
            ahse_params = self.model_config.get('ahse', {}).copy()

            # 2. Retirer 'model_names' : non accepté par AHSEMultiHorizon.__init__
            ahse_params.pop('model_names', None)

            # 3. Initialiser le sélecteur
            ahse_selector = AHSEMultiHorizon(**ahse_params)

            # Propager kt_feature_idx pour que le GatingNetwork route sur σ(kt)
            # et non sur la feature 0 (Wind_Speed) — bug #12
            kt_idx = data.get('kt_feature_idx')
            if kt_idx is None:
                logger.error("❌ kt_feature_idx absent des données — gating impossible sur kt !")
            else:
                logger.info(f"   ℹ️  kt est à l'index {kt_idx} dans la matrice de features")
                ahse_selector.gating.kt_feature_idx = kt_idx

            # 4. SÉLECTION sur VAL set
            #    y_val ne contient que des séquences diurnes (filtrage appliqué
            #    dans preprocessor.py — kt > 0.05 dans l'horizon). Pas de patch ici.
            _, info = ahse_selector.select_and_combine(
                predictions=self.predictions_val,  # prédictions val (diurnes)
                y_true=y_val,                      # labels val (diurnes)
                X_test=X_val                       # features val pour le gating
            )
            logger.info(f"   🔍 Stratégie sélectionnée (sur val) : {info['method']}")
            logger.info(f"   📋 Modèles retenus : {info.get('selected_models', [])}")

            # 5. PRÉDICTION sur TEST set (y_test non exposé à la sélection)
            ahse_preds_test = ahse_selector.predict(
                predictions=self.predictions,  # prédictions sur X_test
                X_test=X_test                  # features test pour le gating
            )

            # 6. Enregistrer les résultats — the original single-strategy selector is kept
            #    for comparison as AHSE_global (note: it ranks models on all validation
            #    samples, night included).
            self.predictions['AHSE_global'] = ahse_preds_test
            self.trained_models['AHSE_global'] = {'model': ahse_selector, 'history': None, 'time': 0}
            joblib.dump(ahse_selector, self.models_dir / f"AHSE_global_{horizon}.joblib")

            # 7. AHSE (block version): per-horizon-block honest selection on daytime
            #    validation targets, statistical guard (bootstrap over days).
            steps_per_hour = int(pd.Timedelta('1h') / pd.Timedelta(horizon))
            blocks = default_blocks(y_val.shape[1], steps_per_hour)
            starts_val = data['target_start_rows']['val']
            day_ids_val = pd.factorize(pd.DatetimeIndex(data['time_index'][starts_val]).date)[0]
            mask_val = data.get('daytime_mask_val')
            if mask_val is None:
                mask_val = y_val > 0
            block_selector = HonestBlockSelector(blocks).fit(
                {m: p for m, p in self.predictions_val.items() if m != 'AHSE_global'},
                y_val, mask_val, day_ids_val)
            self.predictions['AHSE'] = block_selector.predict(self.predictions)
            self.trained_models['AHSE'] = {'model': block_selector, 'history': None, 'time': 0}
            logger.info(f"   🧱 AHSE (blocks) reference: {block_selector.reference}")
            for name, dec in block_selector.decisions.items():
                logger.info(f"      {name}: {dec['chosen']} (LCB gain {dec['lcb_gain']:.4f})")
            with open(self.tables_dir / f"ahse_block_decisions_{horizon}.json", "w") as fh:
                json.dump({'reference': block_selector.reference, 'decisions': block_selector.decisions},
                          fh, indent=1, default=float)
            joblib.dump(block_selector, self.models_dir / f"AHSE_{horizon}.joblib")
            logger.info(f"   ✅ AHSE terminé — évaluation finale sur test set propre")
        
        # 1. Évaluer (Retourne un dictionnaire avec Global et One-Step)
        eval_dfs = self.evaluate(y_test, self.predictions, daytime_mask)
        
        # 2. Sauvegarder les deux tables séparément
        eval_dfs['global'].to_csv(self.tables_dir / f"metrics_global_{horizon}.csv", index=False)
        eval_dfs['onestep'].to_csv(self.tables_dir / f"metrics_onestep_{horizon}.csv", index=False)

        # Prediction cache made of plain numeric arrays only (no Python objects), used by
        # post-hoc analyses such as per-horizon selection or conformal calibration.
        cache = {'y_val': y_val, 'y_test': y_test,
                 'daytime_mask_test': daytime_mask,
                 'daytime_mask_val': data.get('daytime_mask_val'),
                 'kt_feature_idx': np.array(-1 if data.get('kt_feature_idx') is None
                                            else data['kt_feature_idx'])}
        for split, starts in (data.get('target_start_rows') or {}).items():
            cache[f'target_start_rows_{split}'] = starts
        cache.update({f'val__{m}': p for m, p in self.predictions_val.items()})
        cache.update({f'test__{m}': p for m, p in self.predictions.items()})
        np.savez_compressed(self.tables_dir / f"predictions_{horizon}.npz",
                            **{k: v for k, v in cache.items() if v is not None})

        logger.info(f"\n📄 Fichiers créés : metrics_global_{horizon}.csv et metrics_onestep_{horizon}.csv")
        
        # 3. Graphiques (utilisent les scores globaux)
        self._generate_plots(horizon, y_test, daytime_mask, eval_dfs['global'])
        self._analyze_horizon_degradation(y_test, horizon)

        # ✅ 4. Lancer la validation statistique (Résidus, DM Test, Feature Importance)
        # Elle est déjà intégrée à la classe, donc pas besoin d'objet externe
        self._run_statistical_validation(y_test)
        
        # ✅ 5. Sauvegarde des décisions AHSE pour la Figure 7 de thesis_visualizations.py
        if 'AHSE' in self.predictions and 'info' in locals():
            pd.DataFrame(info.get('decisions', [])).to_csv(
                self.tables_dir / f"gating_decisions_{horizon}.csv", index=False
            )
        
        logger.info("\n" + "=" * 60 + "\n✅ PIPELINE TERMINÉ\n" + "=" * 60)
        
        # ✅ 6. L'UNIQUE retour de la fonction
        return {
            'metrics_global': eval_dfs['global'],
            'metrics_onestep': eval_dfs['onestep'],
            'predictions': self.predictions,
            'data': data
        }
        
    def _generate_plots(
        self,
        horizon: str,
        y_test: np.ndarray,
        daytime_mask: np.ndarray,
        metrics_df: pd.DataFrame
    ):
        """Generate publication-quality plots."""
        import matplotlib.pyplot as plt

        # Modèles de référence exclus des figures comparatives
        # (conservés dans les tableaux métriques pour FSS et comparaison chiffrée)
        BASELINE_MODELS = {'Persistence', 'SmartPersistence'}

        logger.info("\n📊 GÉNÉRATION DES FIGURES")

        # 1. Model comparison heatmap (sans les baselines pour préserver l'échelle couleur)
        metrics_ml = metrics_df[~metrics_df['Model'].isin(BASELINE_MODELS)].copy()
        fig = plot_model_comparison_heatmap(metrics_ml)
        save_figure_for_publication(
            fig, f"model_comparison_{horizon}", str(self.figures_dir)
        )
        plt.close(fig)

        # 1b. Heatmap séparée incluant les baselines (table complète pour l'annexe)
        fig_all = plot_model_comparison_heatmap(metrics_df)
        save_figure_for_publication(
            fig_all, f"model_comparison_all_{horizon}", str(self.figures_dir)
        )
        plt.close(fig_all)

        # 2. Forecast comparison — ML + AHSE seulement (3 derniers jours)
        if hasattr(self, 'data') and 'timestamps' in self.data:
            timestamps = self.data['timestamps'][-len(y_test):]

            # Moyenne sur l'horizon pour la représentation temporelle
            y_actual_mean = y_test.mean(axis=1) if y_test.ndim > 1 else y_test

            # Exclure les baselines du graphique temporel
            preds_mean = {
                k: v.mean(axis=1) if v.ndim > 1 else v
                for k, v in self.predictions.items()
                if k not in BASELINE_MODELS
            }

            fig = plot_forecast_comparison(
                timestamps=timestamps,
                y_actual=y_actual_mean,
                predictions=preds_mean,
                highlight_days=3
            )
            save_figure_for_publication(
                fig, f"forecast_comparison_{horizon}", str(self.figures_dir)
            )
            plt.close(fig)
    
    def _analyze_horizon_degradation(self, y_test: np.ndarray, horizon: str):
        """Analyze error degradation across forecast horizon."""
        if y_test.ndim == 1:
            logger.warning("   ⚠️ Single-step predictions, skipping horizon analysis")
            return
        
        logger.info("\n📈 ANALYSE DÉGRADATION PAR HORIZON")
        
        analyzer = HorizonErrorAnalyzer()
        
        BASELINE_MODELS = {'Persistence', 'SmartPersistence'}
        horizon_errors = {}
        for model_name, preds in self.predictions.items():
            if model_name in BASELINE_MODELS:
                continue  # Baselines omises — courbes de dégradation ML uniquement
            if preds.ndim > 1 and preds.shape[1] == y_test.shape[1]:
                errors = analyzer.compute_horizon_rmse(y_test, preds)
                horizon_errors[model_name] = errors
        
        if horizon_errors:
            # Get resolution from config
            res_map = {'5min': 5, '15min': 15, '30min': 30, '1h': 60, '2h': 120}
            resolution = res_map.get(horizon, 15)
            
            fig = plot_horizon_degradation(
                horizon_errors,
                metric='RMSE',
                resolution_minutes=resolution,
                title=f"Dégradation RMSE - Horizon {horizon}"
            )
            save_figure_for_publication(
                fig, f"horizon_degradation_{horizon}", str(self.figures_dir)
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

    def _run_statistical_validation(self, y_test: np.ndarray):
        """Version complète intégrant TOUTES les analyses du fichier spécialisé."""
        logger.info("\n🧪 VALIDATION STATISTIQUE COMPLÈTE")
        
        # 1. Analyse des résidus et Sauvegarde CSV
        target_model = 'AHSE' if 'AHSE' in self.predictions else list(self.predictions.keys())[0]
        y_pred = self.predictions[target_model]
        residuals = y_test.flatten() - y_pred.flatten()
        
        res_stats = {
            'Model': target_model,
            'Mean': np.mean(residuals),
            'Std': np.std(residuals),
            'Skewness': stats.skew(residuals),
            'Kurtosis': stats.kurtosis(residuals)
        }
        ks_stat, p_val_ks = stats.kstest((residuals - np.mean(residuals))/np.std(residuals), 'norm')
        res_stats['KS_p_value'] = p_val_ks
        
        # Sauvegarde CSV
        pd.DataFrame([res_stats]).to_csv(self.tables_dir / f"stats_residuals_{target_model}.csv", index=False)
        logger.info(f"   📊 Résidus ({target_model}): Skewness={res_stats['Skewness']:.4f}, KS-p={p_val_ks:.2e}")

        # 2. Matrice Diebold-Mariano complète (HAC Newey-West + BH correction)
        if len(self.predictions) > 1:
            from src.evaluation.statistical_analysis import DieboldMarianoTest

            # hac_nlags : 96 pour données 15-min (cycle diurne), 24 pour 1h
            hac_nlags = 96 if getattr(self, 'horizon', '15min') == '15min' else 24

            try:
                dm_matrix, p_matrix, p_adj_matrix = DieboldMarianoTest.run_full_dm_matrix(
                    predictions=self.predictions,
                    y_true=y_test,
                    loss_type='squared',
                    hac_nlags=hac_nlags,
                    alpha=0.05,
                )

                # Sauvegarde CSV — avec suffixe horizon pour éviter l'écrasement
                hz = getattr(self, 'horizon', '15min')
                dm_matrix.to_csv(self.tables_dir / f"dm_matrix_{hz}.csv")
                p_matrix.to_csv(self.tables_dir / f"dm_pvalues_{hz}.csv")
                p_adj_matrix.to_csv(self.tables_dir / f"dm_pvalues_bh_{hz}.csv")
                logger.info(f"   ⚖️  Matrice DM sauvegardée : dm_matrix_{hz}.csv (HAC NW, BH corrigé)")

                # Heatmap
                try:
                    fig_path = self.figures_dir / f"dm_heatmap_{hz}.pdf"
                    DieboldMarianoTest.plot_dm_heatmap(
                        dm_matrix, p_adj_matrix,
                        alpha=0.05,
                        save_path=str(fig_path),
                        title=f"Matrice DM — HAC NW (lags={hac_nlags}), BH α=0.05",
                    )
                    import matplotlib.pyplot as _plt
                    _plt.close("all")
                    logger.info("   🖼️  Heatmap DM sauvegardée : %s", fig_path)
                except Exception as e_plot:
                    logger.warning("   Heatmap DM échouée : %s", e_plot)

                # Résumé AHSE vs tous (si AHSE présent)
                if 'AHSE' in self.predictions:
                    ahse_row = dm_matrix.loc['AHSE'] if 'AHSE' in dm_matrix.index else None
                    if ahse_row is not None:
                        logger.info("   AHSE DM stats vs autres : %s",
                                    ahse_row.drop('AHSE', errors='ignore').to_dict())

            except Exception as e_dm:
                logger.warning("   Matrice DM échouée : %s", e_dm)

        # 3. Graphiques Feature Importance (XAI)
        for name in ['LightGBM', 'XGBoost']:
            if name in self.trained_models and hasattr(self.trained_models[name]['model'], 'feature_importances_'):
                importances = self.trained_models[name]['model'].feature_importances_
                indices = np.argsort(importances)[-10:] # Top 10
                
                plt.figure(figsize=(10, 6))
                plt.title(f"Feature Importance - {name}")
                plt.barh(range(len(indices)), importances[indices])
                names = self._tree_feature_names()
                if len(names) != len(importances):
                    names = [f"f{i}" for i in range(len(importances))]
                plt.yticks(range(len(indices)), [names[i] for i in indices])
                plt.tight_layout()
                plt.savefig(self.figures_dir / f"importance_{name}.png")
                plt.close()
                logger.info(f"   🌳 Graphique d'importance sauvé pour {name}")
    
    def save_models(self, prefix: str = ''):
        """Save trained models with specific handling for AHSE."""
        for model_name, result in self.trained_models.items():
            if result['model'] is not None:
                path_base = self.models_dir / f"{prefix}{model_name}"
                
                # Cas 1: Modèles d'arbres ou AHSE (Format Joblib)
                if model_name in ['LightGBM', 'XGBoost', 'AHSE']:
                    path = path_base.with_suffix('.joblib')
                    joblib.dump(result['model'], path)
                    logger.info(f"   💾 Saved (joblib): {path}")
                
                # Cas 2: Réseaux de neurones (Format PyTorch)
                elif model_name not in ['Persistence', 'SmartPersistence']:
                    path = path_base.with_suffix('.pt')
                    torch.save(result['model'].state_dict(), path)
                    logger.info(f"   💾 Saved (torch): {path}")


DATASET_CONFIGS = {
    'yulara_neighbours': 'configs/data_config_yulara_neighbours.yaml',
    'nist_pvdaq': 'configs/data_config_nist.yaml',
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PV Forecasting SOTA Training Pipeline'
    )
    
    parser.add_argument(
        '--horizon', '-hz',
        type=str,
        default='15min',
        choices=['5min', '15min', '30min', '1h', '2h'],
        help='Forecast horizon (default: 15min)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='yulara_neighbours',
        choices=sorted(DATASET_CONFIGS),
        help='Dataset: data/processed/<dataset>_<horizon>.csv and its configuration'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['all'],
        help='Models to train (default: all)'
    )
    
    parser.add_argument(
        '--data-config',
        type=str,
        default=None,
        help='Data configuration (default: configs/data_config_<site>.yaml of --dataset)'
    )
    
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/model_params.yaml',
        help='Path to model config'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,  # Will be set based on --dataset
        help='Output directory (default: results/<dataset>/)'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Processed CSV (default: <processed_dir>/<results-dir name>_<horizon>.csv)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='Last date (YYYY-MM-DD) kept, e.g. the end of the GFS archive'
    )

    parser.add_argument(
        '--gfs',
        nargs=2,
        default=None,
        metavar=('DSWRF_CSV', 'TCDC_CSV'),
        help='GFS archives (scripts/gdex_batch.py) used as day-ahead covariates'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training, only evaluate'
    )
    
    parser.add_argument(
        '--save-models',
        action='store_true',
        help='Save trained models'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # ==========================================================================
    # REPRODUCIBILITY: Set seed FIRST
    # ==========================================================================
    seed = args.seed
    init_reproducibility(seed, verbose=True)
    
    # Defaults derived from the dataset
    results_dir = args.results_dir or f'results/{args.dataset}'
    args.data_config = args.data_config or DATASET_CONFIGS[args.dataset]
    args.data_file = args.data_file or f'data/processed/{args.dataset}_{args.horizon}.csv'
    
    # Initialize pipeline
    pipeline = PVForecastingPipeline(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        results_dir=results_dir,
        data_file=args.data_file,
        end=args.end,
        gfs_files=tuple(args.gfs) if args.gfs else None,
        seed=seed
    )
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        horizon=args.horizon,
        models=args.models,
        skip_training=args.skip_training
    )
    
    # Save models if requested
    if args.save_models:
        pipeline.save_models(prefix=f"{args.horizon}_")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES PERFORMANCES GLOBALES (24h)")
    print("=" * 60)
    print(results['metrics_global'].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🎯 RÉSUMÉ DES PERFORMANCES ONE-STEP (t+1)")
    print("=" * 60)
    print(results['metrics_onestep'].to_string(index=False))
    
    return results


if __name__ == '__main__':
    main()
