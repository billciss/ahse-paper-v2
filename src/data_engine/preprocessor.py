"""
Préprocesseur Multi-Horizon pour Prévision PV
==============================================
Génère des séquences X avec lookback et des cibles Y de taille H
pour la prévision multi-step directe.

Auteur: Master Thesis - Electrical Engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Dict, Any, Optional, List
import logging
import yaml

from .pv_physics import ClearSkyCalculator

logger = logging.getLogger(__name__)


class MultiHorizonPreprocessor:
    """
    Préprocesseur pour prévision multi-horizon.
    
    Caractéristiques clés:
    1. Prédiction de kt (indice de clarté) au lieu de la puissance brute
    2. Sortie vectorielle de taille H (multi-output)
    3. Support pour différentes résolutions temporelles
    4. Alignement correct des données pour l'évaluation
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        latitude: float = -25.24,
        longitude: float = 131.04,
        altitude: float = 492,
        timezone: str = "Australia/Darwin",
        scaler_type: str = "standard",
        use_clear_sky_index: bool = True,
        night_threshold: float = 10.0,
        kt_epsilon: float = 1.0,
        kt_clip_max: float = 1.5
    ):
        """
        Args:
            config_path: Chemin vers configs/data_config_<site>.yaml (prioritaire)
            latitude, longitude, etc.: Paramètres manuels si pas de config
        """
        # Charger config si fournie
        if config_path:
            self._load_config(config_path)
        else:
            self.latitude = latitude
            self.longitude = longitude
            self.altitude = altitude
            self.timezone = timezone
            self.scaler_type = scaler_type
            self.use_kt = use_clear_sky_index
            self.night_threshold = night_threshold
            self.kt_epsilon = kt_epsilon
            self.kt_clip_max = kt_clip_max
        
        # Initialiser le calculateur de ciel clair
        self.clear_sky_calc = ClearSkyCalculator(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            timezone=self.timezone,
            array_tilt=getattr(self, 'array_tilt', None),
            array_azimuth=getattr(self, 'array_azimuth', None),
            power_model=getattr(self, 'clear_sky_power_model', 'poa_clear')
        )

        # Scalers
        self.feature_scaler = None
        self.target_scaler = None

        # Métadonnées
        self.feature_columns = []
        self.kt_feature_idx = None   # Colonne de kt dans la matrice de features
        self.index_info = {}
        self.clear_sky_power = None  # Pour la recomposition
    
    def _load_config(self, config_path: str):
        """Charge la configuration depuis YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        loc = config['location']
        self.latitude = loc['latitude']
        self.longitude = loc['longitude']
        self.altitude = loc.get('altitude', 0)
        self.timezone = loc['timezone']
        self.array_tilt = loc.get('array_tilt')
        self.array_azimuth = loc.get('array_azimuth')

        preproc = config.get('preprocessing', {})
        self.clear_sky_power_model = preproc.get('clear_sky_power_model', 'poa_clear')
        self.scaler_type = preproc.get('scaler_type', 'standard')
        self.use_kt = preproc.get('use_clear_sky_index', True)
        self.night_threshold = preproc.get('night_threshold_ghi', 10.0)
        self.kt_epsilon = preproc.get('kt_epsilon', 1.0)
        self.kt_clip_max = preproc.get('kt_clip_max', 1.5)
        # "exclude": sequences touching a gap longer than max_interpolation_gap are dropped;
        # "interpolate": legacy behaviour (every gap linearly interpolated)
        self.gap_policy = preproc.get('gap_policy', 'exclude')
        self.max_interpolation_gap = preproc.get('max_interpolation_gap', '1h')

        # Validation design. "chronological": train | val | test in time order (default).
        # "interleaved": the test set is still the final block, but inside the preceding
        # train+val span every `val_every`-th block of `val_block_days` days is validation, so
        # that validation covers every season; sequences whose window (lookback + horizon)
        # mixes training and validation rows are dropped (embargo).
        split = config.get('data_split', {})
        self.validation_mode = split.get('validation', 'chronological')
        self.val_block_days = int(split.get('val_block_days', 7))
        self.val_every = int(split.get('val_every', 4))

        logger.info(f"✅ Configuration chargée depuis {config_path}")
    
    def prepare_multi_horizon_data(
        self,
        df: pd.DataFrame,
        horizon: str = "15min",
        lookback_window: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        target_column: str = "Active_Power",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Dict[str, Any]:
        """
        Prépare les données pour la prévision multi-horizon.
        
        Args:
            df: DataFrame avec index DatetimeIndex
            horizon: "5min", "15min", "30min", "1h", "2h"
            lookback_window: Taille de la fenêtre d'entrée (auto si None)
            forecast_horizon: Taille du vecteur de sortie H (auto si None)
            target_column: Colonne cible
            train_ratio, val_ratio: Proportions des splits
        
        Returns:
            Dict avec X_train, y_train, etc., scaler, index_info
        """
        logger.info("=" * 60)
        logger.info("🔧 PRÉPARATION MULTI-HORIZON")
        logger.info("=" * 60)
        
        # Paramètres par défaut selon l'horizon
        horizon_params = {
            '5min': {'lookback': 288, 'horizon': 12},
            '15min': {'lookback': 96, 'horizon': 96},
            '30min': {'lookback': 48, 'horizon': 48},
            '1h': {'lookback': 24, 'horizon': 24},
            '2h': {'lookback': 12, 'horizon': 12}
        }
        
        params = horizon_params.get(horizon, horizon_params['15min'])
        lookback = lookback_window or params['lookback']
        H = forecast_horizon or params['horizon']
        
        logger.info(f"📊 Horizon: {horizon}")
        logger.info(f"   Lookback: {lookback} pas")
        logger.info(f"   Forecast horizon H: {H} pas")
        
        # 1. Valider et préparer les données
        df = self._validate_and_prepare(df, target_column)

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        interleaved = self.validation_mode == 'interleaved'
        if interleaved:
            step = pd.Series(df.index[:1000]).diff().median()      # grid step
            block_rows = int(pd.Timedelta(days=self.val_block_days) / step)
            is_val_row = np.zeros(n, dtype=bool)
            dev = np.arange(val_end)
            is_val_row[:val_end] = (dev // block_rows) % self.val_every == self.val_every - 1
            pcs_fit_mask = (np.arange(n) < val_end) & ~is_val_row
            logger.info("🔀 Interleaved validation: every %d-th block of %d days before the test "
                        "set (%.1f%% of those rows)", self.val_every, self.val_block_days,
                        100 * is_val_row[:val_end].mean())
        else:
            pcs_fit_mask = None
        
        # 2. Ajouter les features physiques
        if self.use_kt:
            logger.info("🔬 Mode Clear Sky Index (kt) activé")
            df = self.clear_sky_calc.add_physics_features(
                df, 
                power_column=target_column,
                epsilon=self.kt_epsilon,
                kt_max=self.kt_clip_max,
                fit_fraction=train_ratio,
                fit_mask=pcs_fit_mask
            )
            # Sauvegarder P_cs pour la recomposition
            self.clear_sky_power = df['clear_sky_power'].copy()
            target_col = 'kt'
        else:
            target_col = target_column
        
        # 3. Séparer features et target
        # NOTE : 'kt' est INCLUS comme feature (valeurs PASSÉES de kt).
        # Prédire kt futur depuis kt passé = autoregression, pas du leakage.
        # Référence : Voyant et al. (2017), Pedro & Coimbra (2012).
        # Cela corrige aussi PersistenceModel (peut utiliser le dernier kt connu).
        feature_cols = [c for c in df.columns if c not in [target_column, 'clear_sky_power']]
        self.feature_columns = feature_cols
        self.kt_feature_idx = feature_cols.index('kt') if 'kt' in feature_cols else None
        
        X_df = df[feature_cols]
        y_series = df[target_col]
        
        logger.info(f"📋 Features ({len(feature_cols)}): {feature_cols[:5]}...")
        
        if interleaved:
            X_train, y_train, X_val, y_val, X_test, y_test, starts = self._interleaved_split(
                X_df.values, y_series.values, is_val_row, val_end, lookback, H)
        else:
            # 4. Split temporel
        
            X_train_raw = X_df.iloc[:train_end].values
            X_val_raw = X_df.iloc[train_end:val_end].values
            X_test_raw = X_df.iloc[val_end:].values
        
            y_train_raw = y_series.iloc[:train_end].values
            y_val_raw = y_series.iloc[train_end:val_end].values
            y_test_raw = y_series.iloc[val_end:].values
        
            # 5. Normalisation (fit sur train uniquement)
            X_train_scaled, X_val_scaled, X_test_scaled = self._normalize_features(
                X_train_raw, X_val_raw, X_test_raw
            )
        
            # 6. Créer les séquences multi-horizon
            X_train, y_train = self._create_multi_horizon_sequences(
                X_train_scaled, y_train_raw, lookback, H
            )
            X_val, y_val = self._create_multi_horizon_sequences(
                X_val_scaled, y_val_raw, lookback, H
            )
            X_test, y_test = self._create_multi_horizon_sequences(
                X_test_scaled, y_test_raw, lookback, H
            )

            # Row (in the prepared frame) of the first target step of every sequence, filtered
            # together with the sequences so that other models (e.g. foundation models using a
            # longer context) can be evaluated on exactly the same samples.
            starts = [np.arange(len(X_train)) + lookback,
                      np.arange(len(X_val)) + train_end + lookback,
                      np.arange(len(X_test)) + val_end + lookback]

            # 6a. Drop sequences touching long gaps (sequence j uses rows j .. j+lookback+H-1)
            if getattr(self, 'gap_policy', 'exclude') == 'exclude':
                gap = getattr(self, 'long_gap_rows', np.zeros(n, dtype=bool))
                keep = [self._windows_without_gaps(gap[a:b], lookback + H)
                        for a, b in [(0, train_end), (train_end, val_end), (val_end, n)]]
                X_train, y_train = X_train[keep[0]], y_train[keep[0]]
                X_val, y_val = X_val[keep[1]], y_val[keep[1]]
                X_test, y_test = X_test[keep[2]], y_test[keep[2]]
                starts = [s[k] for s, k in zip(starts, keep)]
                logger.info("🕳️  Long gaps (> %s): %d rows; sequences dropped train/val/test = %d/%d/%d",
                            getattr(self, 'max_interpolation_gap', '1h'), int(gap.sum()),
                            int((~keep[0]).sum()), int((~keep[1]).sum()), int((~keep[2]).sum()))

        # 6b. Filtrage diurne — standard en prévision solaire industrielle
        #
        # On conserve uniquement les séquences dont l'horizon de prévision contient
        # AU MOINS UN pas avec une production solaire significative (kt > 0.05).
        # Cela élimine les séquences "24h de nuit pure" (quasi-inexistantes à
        # Yulara, lat -25°) tout en conservant les transitions aube/crépuscule.
        #
        # Justification : Lorenz et al. (2011), Pedro & Coimbra (2012),
        # Wang et al. (2019). Les solutions industrielles (Solargis, ECMWF-PVGIS)
        # n'entraînent pas sur données nocturnes.
        kt_threshold = 0.05
        if self.use_kt:
            train_day = np.any(y_train > kt_threshold, axis=1)
            val_day   = np.any(y_val   > kt_threshold, axis=1)
            test_day  = np.any(y_test  > kt_threshold, axis=1)

            n_train_before = len(X_train)
            X_train, y_train = X_train[train_day], y_train[train_day]
            X_val,   y_val   = X_val[val_day],     y_val[val_day]
            X_test,  y_test  = X_test[test_day],   y_test[test_day]
            starts = [starts[0][train_day], starts[1][val_day], starts[2][test_day]]

            logger.info("☀️  Filtrage diurne (kt > %.2f dans l'horizon) :", kt_threshold)
            logger.info("   Train : %d → %d séq. (%.1f%% retenues)",
                        n_train_before, len(X_train),
                        100 * len(X_train) / max(n_train_before, 1))
            logger.info("   Val   : %d séq. diurnes", len(X_val))
            logger.info("   Test  : %d séq. diurnes", len(X_test))

        # 7. Stocker les métadonnées
        self.index_info = {
            'train_start': 0,
            'train_end': train_end,
            'val_start': train_end,
            'val_end': val_end,
            'test_start': val_end,
            'test_end': n,
            'lookback_window': lookback,
            'forecast_horizon': H,
            'test_start_after_window': val_end + lookback,
            'original_index': df.index,
            'target_column': target_col,
            'use_kt': self.use_kt
        }
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("✅ DONNÉES PRÊTES")
        logger.info(f"   X_train: {X_train.shape} (samples, lookback, features)")
        logger.info(f"   y_train: {y_train.shape} (samples, horizon)")
        logger.info(f"   X_val:   {X_val.shape}")
        logger.info(f"   X_test:  {X_test.shape}")
        logger.info("=" * 60)
        
        # Masque diurne par-pas pour l'évaluation : kt=0 la nuit par construction
        # (pv_physics.py impose kt[night_mask]=0), donc y_test>0 <=> diurne
        daytime_mask_test = (y_test > 0.0)   # shape (n_test_samples, horizon)
        daytime_mask_val  = (y_val  > 0.0)   # idem pour le val set

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_scaler': self.feature_scaler,
            'index_info': self.index_info,
            'feature_columns': self.feature_columns,
            'feature_names': self.feature_columns,   # alias for XAI plots
            'clear_sky_power': self.clear_sky_power,
            'kt_feature_idx': self.kt_feature_idx,   # index of kt in feature matrix
            'daytime_mask': daytime_mask_test,        # shape (n_test, H) — bug #14
            'daytime_mask_val': daytime_mask_val,     # shape (n_val, H)
            'target_series': y_series.to_numpy(),     # target over the whole prepared frame
            'time_index': df.index,
            'target_start_rows': {'train': starts[0], 'val': starts[1], 'test': starts[2]},
        }
    
    def _validate_and_prepare(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Valide et nettoie les données."""
        df = df.copy()
        
        # Index datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ['timestamp', 'Datetime', 'datetime', 'date', 'ds', 'time']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
        
        # Vérifier la target
        if target_column not in df.columns:
            raise ValueError(f"❌ Colonne '{target_column}' non trouvée!")
        
        # Garder colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols]
        
        # Corriger valeurs négatives
        if (df[target_column] < 0).any():
            df.loc[df[target_column] < 0, target_column] = 0
        
        # Rows inside gaps longer than max_interpolation_gap: they are still interpolated so
        # that arrays stay finite, but every sequence touching them is dropped later when
        # gap_policy == "exclude".
        self.long_gap_rows = self._long_gap_rows(df)

        # Interpoler NaN
        df = df.interpolate(method='time').bfill().ffill()
        
        logger.info(f"✅ Données validées: {len(df):,} lignes, {len(df.columns)} colonnes")
        
        return df
    
    def _long_gap_rows(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean mask of rows belonging to a NaN run longer than max_interpolation_gap."""
        na = df.isna().any(axis=1)
        if not na.any():
            return np.zeros(len(df), dtype=bool)
        step = pd.Series(df.index).diff().median()
        limit = max(1, int(pd.Timedelta(getattr(self, 'max_interpolation_gap', '1h')) / step))
        run_len = na.groupby((~na).cumsum()).transform('sum').where(na, 0)
        return (run_len > limit).to_numpy()

    @staticmethod
    def _windows_without_gaps(gap: np.ndarray, window: int) -> np.ndarray:
        """For each sequence start j, True if rows j .. j+window-1 contain no long-gap row."""
        c = np.concatenate([[0], np.cumsum(gap.astype(int))])
        n_seq = len(gap) - window + 1
        if n_seq <= 0:
            return np.zeros(0, dtype=bool)
        return (c[window:window + n_seq] - c[:n_seq]) == 0

    def _normalize_features(
        self,
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalise les features (fit sur train uniquement)."""
        if self.scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
        else:
            self.feature_scaler = StandardScaler()
        
        self.feature_scaler.fit(X_train)
        
        return (
            self.feature_scaler.transform(X_train),
            self.feature_scaler.transform(X_val),
            self.feature_scaler.transform(X_test)
        )
    
    def _interleaved_split(self, X, y, is_val_row, val_end, lookback, H):
        """Train/val sequences from interleaved blocks before val_end, test after it."""
        n = len(X)
        train_rows = np.zeros(n, dtype=bool)
        train_rows[:val_end] = ~is_val_row[:val_end]
        self.feature_scaler = {'minmax': MinMaxScaler, 'robust': RobustScaler}.get(
            self.scaler_type, StandardScaler)()
        self.feature_scaler.fit(X[train_rows])
        X_dev = self.feature_scaler.transform(X[:val_end])
        X_test_s = self.feature_scaler.transform(X[val_end:])

        W = lookback + H
        X_d, y_d = self._create_multi_horizon_sequences(X_dev, y[:val_end], lookback, H)
        c = np.concatenate([[0], np.cumsum(is_val_row[:val_end].astype(int))])
        v = c[W:W + len(X_d)] - c[:len(X_d)]           # validation rows in each window
        gap = getattr(self, 'long_gap_rows', np.zeros(n, dtype=bool))
        ok_dev = (self._windows_without_gaps(gap[:val_end], W)
                  if getattr(self, 'gap_policy', 'exclude') == 'exclude' else np.ones(len(X_d), bool))
        tr, va = (v == 0) & ok_dev, (v == W) & ok_dev
        X_te, y_te = self._create_multi_horizon_sequences(X_test_s, y[val_end:], lookback, H)
        ok_te = (self._windows_without_gaps(gap[val_end:], W)
                 if getattr(self, 'gap_policy', 'exclude') == 'exclude' else np.ones(len(X_te), bool))
        j = np.arange(len(X_d)) + lookback
        starts = [j[tr], j[va], (np.arange(len(X_te)) + val_end + lookback)[ok_te]]
        logger.info("🔀 Interleaved split: train %d, val %d, embargo-dropped %d, test %d sequences",
                    int(tr.sum()), int(va.sum()), int((~tr & ~va).sum()), int(ok_te.sum()))
        return X_d[tr], y_d[tr], X_d[va], y_d[va], X_te[ok_te], y_te[ok_te], starts

    def _create_multi_horizon_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lookback: int,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des séquences avec sortie multi-horizon.
        
        Pour chaque point t:
        - X[t] = [X[t-lookback], ..., X[t-1]]  shape: (lookback, n_features)
        - y[t] = [y[t], y[t+1], ..., y[t+horizon-1]]  shape: (horizon,)
        """
        X_seq = []
        y_seq = []
        
        # On a besoin de lookback en arrière et horizon en avant
        for i in range(lookback, len(X) - horizon + 1):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i:i + horizon])
        
        return np.array(X_seq), np.array(y_seq)
    
    def recompose_power(
        self,
        kt_predictions: np.ndarray,
        test_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Recompose la puissance réelle à partir des prédictions de kt.
        
        P_output = kt_pred × P_clearsky
        
        Args:
            kt_predictions: Prédictions de kt shape (n_samples, horizon)
            test_indices: Indices dans l'index original (optionnel)
        
        Returns:
            Puissance recomposée en Watts
        """
        if not self.use_kt:
            return kt_predictions
        
        if self.clear_sky_power is None:
            logger.warning("⚠️ P_clearsky non disponible, retour de kt brut")
            return kt_predictions
        
        # Récupérer P_cs pour les indices de test
        test_start = self.index_info['test_start_after_window']
        horizon = self.index_info['forecast_horizon']
        
        n_samples = kt_predictions.shape[0]
        p_recomposed = np.zeros_like(kt_predictions)
        
        for i in range(n_samples):
            idx_start = test_start + i
            idx_end = idx_start + horizon
            
            if idx_end <= len(self.clear_sky_power):
                p_cs = self.clear_sky_power.iloc[idx_start:idx_end].values
                p_recomposed[i] = kt_predictions[i] * p_cs
            else:
                # Fallback si on dépasse
                p_recomposed[i] = kt_predictions[i] * self.clear_sky_power.iloc[-horizon:].values
        
        return p_recomposed
    
    def get_daytime_mask(
        self,
        test_size: int,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Retourne un masque booléen pour les données diurnes.
        
        Shape: (n_samples, horizon) - True si GHI > seuil
        """
        threshold = threshold or self.night_threshold
        test_start = self.index_info['test_start_after_window']
        horizon = self.index_info['forecast_horizon']
        
        # Utiliser clear_sky_ghi si disponible, sinon clear_sky_power
        if self.clear_sky_power is not None:
            reference = self.clear_sky_power
        else:
            return np.ones((test_size, horizon), dtype=bool)
        
        mask = np.zeros((test_size, horizon), dtype=bool)
        
        for i in range(test_size):
            idx_start = test_start + i
            idx_end = idx_start + horizon
            
            if idx_end <= len(reference):
                mask[i] = reference.iloc[idx_start:idx_end].values > threshold
        
        return mask


def load_and_prepare_multi_horizon(
    data_path: str,
    config_path: str = "configs/data_config_yulara_neighbours.yaml",
    horizon: str = "15min",
    end: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour charger et préparer les données.
    
    Usage:
        data = load_and_prepare_multi_horizon(
            "data/processed/data_15min.csv",
            config_path="configs/data_config_yulara_neighbours.yaml",
            horizon="15min"
        )
        X_train = data['X_train']
        y_train = data['y_train']
    """
    import yaml
    
    # Charger config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Charger données
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    if end is not None:
        # keep whole days up to `end` (e.g. the last date covered by NWP archives)
        last = pd.Timestamp(end) + pd.Timedelta(days=1)
        df = df[df.index < (last.tz_localize(df.index.tz) if df.index.tz is not None else last)]
    
    # Créer preprocessor et préparer
    preprocessor = MultiHorizonPreprocessor(config_path=config_path)
    
    # Trouver les paramètres d'horizon
    horizons = config['forecasting'].get('available_horizons', [])
    horizon_config = next((h for h in horizons if h['name'] == horizon), None)
    
    if horizon_config:
        lookback = horizon_config['lookback']
        forecast_h = horizon_config['horizon']
    else:
        lookback = None
        forecast_h = None
    
    split = config.get('data_split', {})
    
    return preprocessor.prepare_multi_horizon_data(
        df,
        horizon=horizon,
        lookback_window=lookback,
        forecast_horizon=forecast_h,
        target_column=config['preprocessing']['target_column'],
        train_ratio=split.get('train_ratio', 0.6),
        val_ratio=split.get('val_ratio', 0.2)
    )


# Compatibilité avec l'ancien code
class DataPreprocessor(MultiHorizonPreprocessor):
    """Alias pour compatibilité arrière."""
    pass


def load_and_prepare_data(
    data_path: str,
    horizon: str = "15min",
    scaler_type: str = "standard",
    config_path: Optional[str] = None
) -> Tuple:
    """
    Interface de compatibilité avec l'ancien code.
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, scaler, index_info)
    """
    if config_path:
        data = load_and_prepare_multi_horizon(data_path, config_path, horizon)
    else:
        preprocessor = MultiHorizonPreprocessor(scaler_type=scaler_type)
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data = preprocessor.prepare_multi_horizon_data(df, horizon=horizon)
    
    return (
        data['X_train'],
        data['X_val'],
        data['X_test'],
        data['y_train'],
        data['y_val'],
        data['y_test'],
        data['feature_scaler'],
        data['index_info']
    )
