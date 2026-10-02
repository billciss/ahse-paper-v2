"""
AHSE - Adaptive Honest Selection Ensemble
==========================================
Sélection et combinaison intelligente des modèles avec Gating Network.

Évolutions par rapport à la version originale:
1. Gating Network: Sélection dynamique basée sur la variance de kt
2. Support multi-horizon: Combine les prédictions de forme (samples, horizon)
3. Analyse par régime: Utilise LightGBM pour ciel clair, PatchTST pour variable
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.utils.routing_logger import RoutingLogger

logger = logging.getLogger(__name__)


class GatingNetwork:
    """
    Réseau de gating pour sélection dynamique des modèles.
    
    Analyse la variance de l'indice de clarté (kt) pour décider
    quel modèle utiliser:
    - Faible variance → LightGBM (conditions stables)
    - Haute variance → PatchTST/N-HiTS (conditions variables)
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.15,
        lookback_window: int = 12,  # 3h si pas=15min
        model_stable: str = "LightGBM",
        model_variable: str = "PatchTST"
    ):
        self.variance_threshold = variance_threshold
        self.lookback_window = lookback_window
        self.model_stable = model_stable
        self.model_variable = model_variable
        self.decisions = []
        self.kt_feature_idx: Optional[int] = None  # à setter depuis main.py après init
        # Logger optionnel — assigner depuis l'extérieur pour activer le logging par échantillon
        # ex: gating.routing_logger = RoutingLogger(output_path=..., tau=variance_threshold)
        self.routing_logger: Optional["RoutingLogger"] = None
    
    def compute_kt_variance(
        self,
        X: np.ndarray,
        kt_feature_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Calcule la variance locale de kt pour chaque échantillon.
        
        Args:
            X: Features shape (n_samples, seq_len, n_features)
            kt_feature_idx: Index de la feature kt (auto-détecté si None)
        
        Returns:
            Variance par échantillon shape (n_samples,)
        """
        if X.ndim == 2:
            # Déjà aplati, utiliser les dernières valeurs
            n_samples = X.shape[0]
            # Prendre les derniers éléments comme proxy
            variances = np.std(X[:, -self.lookback_window:], axis=1)
        else:
            n_samples = X.shape[0]

            if kt_feature_idx is None:
                kt_feature_idx = self.kt_feature_idx
            if kt_feature_idx is None:
                raise ValueError(
                    "kt_feature_idx must be set. "
                    "Le GatingNetwork ne peut pas calculer σ(kt) sans connaître "
                    "l'index de la colonne kt. Setter ahse_selector.gating.kt_feature_idx "
                    "depuis main.py après la création de l'AHSEMultiHorizon."
                )

            kt_values = X[:, -self.lookback_window:, kt_feature_idx]
            variances = np.std(kt_values, axis=1)
        
        return variances
    
    def select_model(
        self,
        variances: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Sélectionne le modèle pour chaque échantillon.
        
        Returns:
            Prédictions combinées shape (n_samples, horizon)
        """
        n_samples = variances.shape[0]
        
        # Vérifier les modèles disponibles
        if self.model_stable not in predictions:
            available = list(predictions.keys())
            self.model_stable = available[0] if available else None
            logger.warning(f"⚠️ Modèle stable non trouvé, fallback vers {self.model_stable}")
        
        if self.model_variable not in predictions:
            available = list(predictions.keys())
            self.model_variable = available[-1] if available else None
            logger.warning(f"⚠️ Modèle variable non trouvé, fallback vers {self.model_variable}")
        
        pred_stable = predictions.get(self.model_stable)
        pred_variable = predictions.get(self.model_variable)
        
        if pred_stable is None or pred_variable is None:
            # Fallback: moyenne de tous les modèles
            all_preds = list(predictions.values())
            return np.mean(all_preds, axis=0)
        
        # Sélection basée sur la variance
        stable_mask = variances < self.variance_threshold
        
        # Combiner les prédictions
        if pred_stable.ndim == 1:
            combined = np.where(stable_mask, pred_stable, pred_variable)
        else:
            combined = np.where(
                stable_mask[:, np.newaxis],
                pred_stable,
                pred_variable
            )
        
        # Enregistrer les décisions agrégées
        n_stable = stable_mask.sum()
        n_variable = n_samples - n_stable
        self.decisions.append({
            'n_stable': int(n_stable),
            'n_variable': int(n_variable),
            'pct_stable': float(n_stable / n_samples * 100)
        })

        logger.info(f"🎯 Gating: {n_stable} stable ({self.model_stable}), "
                    f"{n_variable} variable ({self.model_variable})")

        # Logging par échantillon (optionnel — activer en assignant self.routing_logger)
        if self.routing_logger is not None:
            self.routing_logger.log_batch(
                variances=variances,
                stable_mask=stable_mask,
            )

        return combined


class AHSEMultiHorizon:
    """
    AHSE adapté pour la prévision multi-horizon.
    
    Stratégies disponibles:
    1. Best Single: Utilise le meilleur modèle uniquement
    2. Average: Moyenne simple des modèles sélectionnés
    3. Stacking: Meta-learner Ridge sur les prédictions
    4. Gating: Sélection dynamique basée sur la variance
    """
    
    def __init__(
        self,
        dominance_threshold: float = 0.30,
        adaptive_gap: float = 0.15,
        min_improvement: float = 1.0,
        gating_threshold: float = 0.15,
        lookback_variance: int = 12,
        stacking_folds: int = 5
    ):
        self.dominance_threshold = dominance_threshold
        self.stacking_folds = stacking_folds
        self.adaptive_gap = adaptive_gap
        self.min_improvement = min_improvement
        
        self.gating = GatingNetwork(
            variance_threshold=gating_threshold,
            lookback_window=lookback_variance
        )
        
        self.selected_models = []
        self.strategy = None
        self.meta_learner = None
    
    def select_and_combine(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        use_gating: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Sélectionne la stratégie d'ensemble et stocke les meta-paramètres.

        WARNING: Appeler cette méthode UNIQUEMENT avec les données de VALIDATION.
        Ne jamais passer y_test ici — cela constituerait une fuite de données.
        Après cet appel, utiliser predict() avec les données de TEST.

        Args:
            predictions: Dict {model_name: preds_val} shape (n_samples_val, horizon)
            y_true: Labels de VALIDATION shape (n_samples_val, horizon)
            X_test: Features de VALIDATION pour le gating (optionnel)
            use_gating: Activer le gating dynamique

        Returns:
            (val_predictions, decision_info) — prédictions sur VAL uniquement,
            jamais utilisées pour l'évaluation finale.
        """
        print("🔍 SELECTION PHASE (on validation set)")
        logger.info("\n" + "=" * 60)
        logger.info("🎯 AHSE MULTI-HORIZON - SÉLECTION D'ENSEMBLE (sur val set)")
        logger.info("=" * 60)
        
        # S'assurer que y_true est 2D
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        # Évaluer tous les modèles
        metrics = {}
        for name, preds in predictions.items():
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            
            # Aplatir pour le calcul des métriques
            mae = mean_absolute_error(y_true.flatten(), preds.flatten())
            rmse = np.sqrt(mean_squared_error(y_true.flatten(), preds.flatten()))
            r2 = r2_score(y_true.flatten(), preds.flatten())
            
            metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
        
        # Trier par MAE
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]['MAE'])
        
        logger.info("\n📊 Classement des modèles (MAE):")
        for name, m in sorted_models:
            logger.info(f"   {name}: MAE={m['MAE']:.4f}, R²={m['R²']:.4f}")
        
        best_name = sorted_models[0][0]
        best_mae = sorted_models[0][1]['MAE']
        best_pred = predictions[best_name]
        
        # Vérifier la dominance
        if len(sorted_models) >= 2:
            gap = (sorted_models[1][1]['MAE'] - best_mae) / (best_mae + 1e-8)
            if gap > self.dominance_threshold:
                logger.info(f"\n⚡ DOMINANCE: {best_name} ({gap:.0%} meilleur)")
                self.strategy = 'Dominant'
                self.selected_models = [best_name]
                return best_pred, {
                    'method': 'Dominant',
                    'selected_models': [best_name],
                    'metrics': metrics[best_name],
                    'improvement': 0
                }
        
        # Sélection adaptative
        selected = [best_name]
        for name, m in sorted_models[1:]:
            gap = (m['MAE'] - best_mae) / (best_mae + 1e-8)
            if gap <= self.adaptive_gap:
                selected.append(name)
        
        logger.info(f"\n📋 Modèles sélectionnés: {selected}")
        
        # Tester les stratégies
        strategies = {}
        
        # 1. Best Single
        strategies['Best_Single'] = {
            'pred': best_pred,
            'mae': best_mae,
            'models': [best_name]
        }
        
        if len(selected) >= 2:
            sel_preds = {m: predictions[m] for m in selected}
            
            # 2. Average
            avg_preds = np.mean(list(sel_preds.values()), axis=0)
            avg_mae = mean_absolute_error(y_true.flatten(), avg_preds.flatten())
            strategies['Average'] = {
                'pred': avg_preds,
                'mae': avg_mae,
                'models': selected
            }
            
            # 3. Stacking (pour chaque horizon)
            if y_true.shape[1] > 1:
                stacked_pred = self._fit_stacking(sel_preds, y_true, selected)
                stacked_mae = mean_absolute_error(y_true.flatten(), stacked_pred.flatten())
                strategies['Stacking'] = {
                    'pred': stacked_pred,
                    'mae': stacked_mae,
                    'models': selected
                }
            
            # 4. Gating (si X_test fourni)
            if use_gating and X_test is not None:
                variances = self.gating.compute_kt_variance(X_test)
                gating_pred = self.gating.select_model(variances, predictions)
                gating_mae = mean_absolute_error(y_true.flatten(), gating_pred.flatten())
                strategies['Gating'] = {
                    'pred': gating_pred,
                    'mae': gating_mae,
                    'models': [self.gating.model_stable, self.gating.model_variable]
                }
        
        # Choisir la meilleure stratégie
        best_strategy = min(strategies.items(), key=lambda x: x[1]['mae'])
        improvement = (best_mae - best_strategy[1]['mae']) / (best_mae + 1e-8) * 100
        
        logger.info(f"\n🏆 Meilleure stratégie: {best_strategy[0]}")
        logger.info(f"   Amélioration: {improvement:.2f}%")
        
        if improvement < self.min_improvement:
            logger.info(f"   → Fallback vers {best_name} (amélioration {improvement:.2f}% < seuil {self.min_improvement}%)")
            # IMPORTANT : écrire self.strategy et self.selected_models pour que predict()
            # retourne le bon modèle (et non la moyenne de tous les modèles).
            self.strategy = 'Fallback'
            self.selected_models = [best_name]
            return best_pred, {
                'method': 'Fallback',
                'selected_models': [best_name],
                'improvement': improvement,
                'metrics': metrics[best_name]
            }
        
        self.strategy = best_strategy[0]
        self.selected_models = best_strategy[1]['models']
        
        return best_strategy[1]['pred'], {
            'method': best_strategy[0],
            'selected_models': self.selected_models,
            'improvement': improvement,
            'all_strategies': {k: v['mae'] for k, v in strategies.items()}
        }
    
    def _fit_stacking(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        selected: List[str]
    ) -> np.ndarray:
        """
        Per-horizon Ridge meta-learners on the VALIDATION set.

        The predictions returned for strategy selection are OUT-OF-FOLD: the
        validation set is split into `stacking_folds` contiguous (unshuffled)
        blocks and each block is predicted by a meta-learner fitted on the other
        blocks. Scoring in-sample predictions would favour Stacking over the
        parameter-free strategies it is compared with. The meta-learners kept for
        predict() on the test set are refitted on the full validation set.
        """
        n_samples, horizon = y_true.shape
        oof_pred = np.zeros((n_samples, horizon))
        folds = KFold(n_splits=self.stacking_folds, shuffle=False)
        self.meta_learners = []

        for h in range(horizon):
            X_stack = np.column_stack([predictions[m][:, h] for m in selected])
            y_h = y_true[:, h]

            for fit_idx, held_idx in folds.split(X_stack):
                meta = Ridge(alpha=1.0).fit(X_stack[fit_idx], y_h[fit_idx])
                oof_pred[held_idx, h] = meta.predict(X_stack[held_idx])

            self.meta_learners.append(Ridge(alpha=1.0).fit(X_stack, y_h))

        return oof_pred

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        X_test: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Génère des prédictions sur le TEST SET avec la stratégie sélectionnée.

        Doit être appelé APRÈS select_and_combine (sur val).
        predictions doit contenir les prédictions des modèles de base sur X_test.
        """
        if self.strategy is None:
            logger.warning("⚠️ AHSE non entraîné, retourne la moyenne")
            return np.mean(list(predictions.values()), axis=0)
        
        if self.strategy == 'Gating' and X_test is not None:
            variances = self.gating.compute_kt_variance(X_test)
            return self.gating.select_model(variances, predictions)
        
        elif self.strategy == 'Stacking' and self.meta_learners:
            n_samples = list(predictions.values())[0].shape[0]
            horizon = len(self.meta_learners)
            result = np.zeros((n_samples, horizon))
            
            for h in range(horizon):
                X_stack = np.column_stack([
                    predictions[m][:, h] for m in self.selected_models
                ])
                result[:, h] = self.meta_learners[h].predict(X_stack)
            
            return result
        
        elif self.strategy == 'Average':
            return np.mean([predictions[m] for m in self.selected_models], axis=0)
        
        else:  # Best_Single, Fallback, Dominant — tous retournent le meilleur modèle seul
            return predictions[self.selected_models[0]]


# Alias pour compatibilité
AHSESelector = AHSEMultiHorizon


def analyze_model_performance_by_regime(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    kt_values: np.ndarray,
    thresholds: Dict[str, Tuple[float, float]] = None
) -> Dict:
    """
    Analyse la performance par régime météo.
    
    Args:
        predictions: Dict des prédictions
        y_true: Valeurs réelles
        kt_values: Indice de clarté correspondant
        thresholds: Seuils pour les régimes
    
    Returns:
        Dict avec les métriques par régime
    """
    if thresholds is None:
        thresholds = {
            'clear_sky': (0.7, 1.5),
            'partly_cloudy': (0.4, 0.7),
            'cloudy': (0.0, 0.4)
        }
    
    results = {}
    
    for regime, (low, high) in thresholds.items():
        mask = (kt_values >= low) & (kt_values < high)
        n_samples = mask.sum()
        
        if n_samples < 50:
            continue
        
        results[regime] = {
            'n_samples': int(n_samples),
            'models': {}
        }
        
        for name, preds in predictions.items():
            y_masked = y_true[mask].flatten()
            pred_masked = preds[mask].flatten()
            
            mae = mean_absolute_error(y_masked, pred_masked)
            rmse = np.sqrt(mean_squared_error(y_masked, pred_masked))
            r2 = r2_score(y_masked, pred_masked)
            
            results[regime]['models'][name] = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R²': float(r2)
            }
    
    return results
