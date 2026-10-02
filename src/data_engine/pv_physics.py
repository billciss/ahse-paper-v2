"""
Module de Physique PV - Clear Sky Index
========================================
Calcule l'indice de clarté (kt) en utilisant pvlib pour 
normaliser la production PV et supprimer les cycles déterministes.

Auteur: Master Thesis - Electrical Engineering
Référence: Ineichen & Perez (2002) pour le modèle de ciel clair
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import warnings
import logging

logger = logging.getLogger(__name__)

# Essayer d'importer pvlib, sinon utiliser un modèle simplifié
try:
    import pvlib
    from pvlib import location as pvlib_location
    from pvlib import irradiance
    from pvlib import solarposition
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    logger.warning("pvlib non disponible. Utilisation du modèle simplifié.")


class ClearSkyCalculator:
    """
    Calculateur de Clear Sky et d'indice de clarté (kt).
    
    L'indice de clarté kt = P_actual / P_clearsky permet de:
    1. Supprimer les cycles jour/nuit déterministes
    2. Normaliser la production pour différentes saisons
    3. Permettre au modèle de se concentrer sur la variabilité météo
    """
    
    def __init__(
        self,
        latitude: float,
        longitude: float,
        altitude: float = 0,
        timezone: str = "UTC",
        model: str = "ineichen",
        array_tilt: Optional[float] = None,
        array_azimuth: Optional[float] = None,
        power_model: str = "poa_clear"
    ):
        """
        Args:
            latitude: Latitude du site (degrés)
            longitude: Longitude du site (degrés)
            altitude: Altitude en mètres
            timezone: Fuseau horaire (ex: "Australia/Darwin", "Asia/Shanghai")
            model: Modèle de ciel clair ("ineichen", "haurwitz", "simplified_solis")
            array_tilt, array_azimuth: géométrie du champ PV (degrés, azimut pvlib: 0=N,
                180=S). Si None, estimée sur les périodes claires de l'entraînement.
            power_model: "poa_clear" (défaut) ou "legacy_correlation"
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        self.model = model
        self.array_tilt = array_tilt
        self.array_azimuth = array_azimuth
        self.power_model = power_model
        self.pcs_params: Optional[Dict[str, float]] = None
        
        if PVLIB_AVAILABLE:
            self.location = pvlib_location.Location(
                latitude=latitude,
                longitude=longitude,
                tz=timezone,
                altitude=altitude
            )
            logger.info(f"☀️ ClearSkyCalculator initialisé avec pvlib")
            logger.info(f"   Site: {latitude:.4f}°, {longitude:.4f}°")
            logger.info(f"   Timezone: {timezone}")
        else:
            logger.info("☀️ ClearSkyCalculator initialisé avec modèle simplifié")
    
    def calculate_solar_position(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calcule la position solaire pour chaque timestamp.
        
        IMPORTANT: L'index temporel DOIT être en timezone locale (Australia/Darwin)
        pour un calcul correct de l'angle zénithal.
        
        Returns:
            DataFrame avec zenith, azimuth, apparent_zenith, etc.
        """
        # S'assurer que l'index est timezone-aware
        if times.tz is None:
            times = times.tz_localize(self.timezone)
            logger.info(f"   ⚠️ Index localisé en {self.timezone}")
        elif str(times.tz) != self.timezone:
            times = times.tz_convert(self.timezone)
            logger.info(f"   ⚠️ Index converti vers {self.timezone}")
        
        if PVLIB_AVAILABLE:
            solar_pos = solarposition.get_solarposition(
                times,
                self.latitude,
                self.longitude,
                altitude=self.altitude
            )
            return solar_pos
        else:
            return self._simple_solar_position(times)
    
    def _simple_solar_position(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        """Modèle simplifié de position solaire."""
        # Jour de l'année
        day_of_year = times.dayofyear.values
        hour = times.hour.values + times.minute.values / 60.0
        
        # Déclinaison solaire (approximation)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Angle horaire
        hour_angle = 15 * (hour - 12)
        
        # Angle zénithal
        lat_rad = np.radians(self.latitude)
        decl_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        
        cos_zenith = (
            np.sin(lat_rad) * np.sin(decl_rad) +
            np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad)
        )
        zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
        
        return pd.DataFrame({
            'zenith': zenith,
            'apparent_zenith': zenith,
            'elevation': 90 - zenith
        }, index=times)
    
    def calculate_clear_sky_ghi(self, times: pd.DatetimeIndex) -> pd.Series:
        """
        Calcule le GHI de ciel clair (irradiance théorique sans nuages).
        
        IMPORTANT: Utilise le fuseau horaire correct (Australia/Darwin = UTC+9:30)
        pour des calculs astronomiques précis.
        
        Returns:
            Series avec le GHI de ciel clair en W/m²
        """
        # Localiser l'index si nécessaire
        if times.tz is None:
            times = times.tz_localize(self.timezone)
        elif str(times.tz) != self.timezone:
            times = times.tz_convert(self.timezone)
        
        if PVLIB_AVAILABLE:
            clear_sky = self.location.get_clearsky(times, model=self.model)
            return clear_sky['ghi']
        else:
            return self._simple_clear_sky_ghi(times)
    
    def _simple_clear_sky_ghi(self, times: pd.DatetimeIndex) -> pd.Series:
        """Modèle simplifié de GHI ciel clair (Haurwitz)."""
        solar_pos = self._simple_solar_position(times)
        zenith = solar_pos['zenith'].values
        
        # Modèle de Haurwitz
        cos_z = np.cos(np.radians(zenith))
        cos_z = np.maximum(cos_z, 0)  # Pas d'irradiance si soleil sous horizon
        
        # GHI = 1098 * cos(z) * exp(-0.057 / cos(z))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ghi = np.where(
                cos_z > 0.05,
                1098 * cos_z * np.exp(-0.057 / cos_z),
                0
            )
        
        return pd.Series(ghi, index=times, name='ghi_clear')
    
    # ------------------------------------------------------------------ clear-sky power
    # Clear-sky PV power envelope P_cs used to define k_t = P / P_cs.
    #
    # "poa_clear" (default): P_cs(t) = min(c * POA_cs(t; tilt, azimuth), P_max), where
    # POA_cs is the Ineichen clear-sky irradiance transposed to the array plane. The
    # scale c, and the array geometry when it is not given in the config, are fitted
    # on clear-sky periods of the TRAINING rows only, detected from MEASURED GHI.
    # "legacy_correlation": the original single P/GHI_cs ratio, kept only to reproduce
    # earlier results; it saturates k_t (see tests/test_clear_sky_power.py).
    CLEAR_ZENITH_MAX = 75.0          # deg; low sun excluded (transposition/cosine errors)
    CLEAR_GHI_BAND = (0.9, 1.1)      # measured GHI / clear-sky GHI
    CLEAR_MAX_STD = 0.05             # 1-h rolling std of that ratio (sub-hourly data only)
    MIN_CLEAR_SAMPLES = 200
    TILT_GRID = np.arange(0.0, 45.1, 5.0)
    AZIMUTH_GRID = np.arange(0.0, 360.0, 30.0)

    def _localize(self, times: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return times.tz_localize(self.timezone) if times.tz is None else times.tz_convert(self.timezone)

    def _poa_clear(self, cs: pd.DataFrame, sp: pd.DataFrame, tilt: float, azimuth: float) -> np.ndarray:
        poa = irradiance.get_total_irradiance(
            tilt, azimuth, sp["apparent_zenith"], sp["azimuth"],
            cs["dni"], cs["ghi"], cs["dhi"])["poa_global"]
        return poa.fillna(0.0).clip(lower=0.0).to_numpy()

    def _clear_mask(self, times, cs, sp, measured_ghi: np.ndarray) -> np.ndarray:
        ghi_cs = cs["ghi"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ghi_cs > 50.0, measured_ghi / ghi_cs, np.nan)
        lo, hi = self.CLEAR_GHI_BAND
        ok = (sp["apparent_zenith"].to_numpy() < self.CLEAR_ZENITH_MAX) & (ratio >= lo) & (ratio <= hi)
        step = pd.Series(times).diff().median()
        if pd.notna(step) and step < pd.Timedelta("1h"):
            std = pd.Series(ratio, index=times).rolling("60min", min_periods=3).std().to_numpy()
            ok &= std <= self.CLEAR_MAX_STD
        return ok

    def fit_clear_sky_power(
        self,
        times: pd.DatetimeIndex,
        actual_power: pd.Series,
        measured_ghi,
        fit_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Fit (tilt, azimuth, scale, P_max) of the clear-sky envelope on training rows."""
        if not PVLIB_AVAILABLE:
            raise ValueError("pvlib is required for the plane-of-array clear-sky envelope.")
        if measured_ghi is None:
            raise ValueError("Measured GHI is required to detect clear-sky periods.")
        times = self._localize(pd.DatetimeIndex(times))
        power = np.asarray(actual_power, dtype=float)
        ghi = np.asarray(measured_ghi, dtype=float)
        fit_mask = np.asarray(fit_mask, dtype=bool)

        p_max = float(np.nanpercentile(power[fit_mask], 99.9))
        cs = self.location.get_clearsky(times, model=self.model)
        sp = self.location.get_solarposition(times)
        clear = (fit_mask & self._clear_mask(times, cs, sp, ghi)
                 & np.isfinite(power) & (power > 0) & (power < 0.98 * p_max))
        n_clear = int(clear.sum())
        if n_clear < self.MIN_CLEAR_SAMPLES:
            raise ValueError(
                f"Only {n_clear} clear-sky training samples found (< {self.MIN_CLEAR_SAMPLES}); "
                "cannot fit the clear-sky power envelope.")

        cs_c, sp_c, p_c = cs[clear], sp[clear], power[clear]
        tilts = [float(self.array_tilt)] if self.array_tilt is not None else self.TILT_GRID
        azimuths = [float(self.array_azimuth)] if self.array_azimuth is not None else self.AZIMUTH_GRID
        best = None
        for tilt in tilts:
            for az in (azimuths if tilt > 0 else azimuths[:1]):
                poa = self._poa_clear(cs_c, sp_c, tilt, az)
                ok = poa > 50.0
                scale = float(np.median(p_c[ok] / poa[ok]))
                err = float(np.median(np.abs(p_c[ok] - scale * poa[ok])) / np.median(p_c[ok]))
                if best is None or err < best["rel_error"]:
                    best = {"tilt": float(tilt), "azimuth": float(az), "scale": scale,
                            "p_max": p_max, "n_clear": n_clear, "rel_error": err}
        self.pcs_params = best
        logger.info(f"   P_cs fitted on {n_clear} clear training samples: tilt={best['tilt']:.0f}°, "
                    f"azimuth={best['azimuth']:.0f}°, scale={best['scale']:.4g}, "
                    f"P_max={best['p_max']:.4g}, median rel. error={best['rel_error']:.3f}")
        return best

    def calculate_clear_sky_power(
        self,
        times: pd.DatetimeIndex,
        actual_power: pd.Series,
        measured_ghi=None,
        fit_mask: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """Clear-sky PV power envelope P_cs (see the notes above)."""
        if self.power_model == "legacy_correlation":
            return self._legacy_clear_sky_power(times, actual_power)
        if self.power_model != "poa_clear":
            raise ValueError(f"Unknown clear-sky power model: {self.power_model}")
        if self.pcs_params is None:
            if fit_mask is None:
                raise ValueError("fit_mask is required: P_cs must be fitted on training rows only.")
            self.fit_clear_sky_power(times, actual_power, measured_ghi, fit_mask)
        prm = self.pcs_params
        loc_times = self._localize(pd.DatetimeIndex(times))
        cs = self.location.get_clearsky(loc_times, model=self.model)
        sp = self.location.get_solarposition(loc_times)
        p_cs = np.minimum(prm["scale"] * self._poa_clear(cs, sp, prm["tilt"], prm["azimuth"]), prm["p_max"])
        index = actual_power.index if isinstance(actual_power, pd.Series) else loc_times
        return pd.Series(p_cs, index=index, name="clear_sky_power")

    def _legacy_clear_sky_power(
        self,
        times: pd.DatetimeIndex,
        actual_power: pd.Series,
        method: str = "correlation"
    ) -> pd.Series:
        """
        LEGACY envelope (single P/GHI_cs ratio). Its "stable period" test is applied to
        the theoretical clear-sky GHI, which is always smooth, so cloudy hours enter the
        ratio; the ratio also ignores array tilt. It saturates k_t and is kept only to
        reproduce earlier results.
        """
        ghi_cs = self.calculate_clear_sky_ghi(times)
        
        if method == "correlation":
            # Trouver le ratio moyen entre P et GHI par ciel clair
            # (jours avec faible variabilité)
            mask_daytime = ghi_cs > 100
            if mask_daytime.sum() > 100:
                # Calculer la variabilité locale
                ghi_std = ghi_cs.rolling(12, center=True, min_periods=1).std()
                ghi_mean = ghi_cs.rolling(12, center=True, min_periods=1).mean()
                cv = (ghi_std / (ghi_mean + 1e-8)).fillna(1)
                
                # Jours stables (faible coefficient de variation)
                mask_stable = (cv < 0.1) & mask_daytime
                
                if mask_stable.sum() > 50:
                    ratio = (actual_power[mask_stable] / ghi_cs[mask_stable]).median()
                else:
                    ratio = actual_power[mask_daytime].max() / ghi_cs[mask_daytime].max()
            else:
                ratio = 1.0
            
            p_cs = ghi_cs * ratio
        else:
            # Méthode du ratio maximum
            mask_daytime = ghi_cs > 50
            if mask_daytime.sum() > 0:
                max_ratio = (actual_power[mask_daytime] / ghi_cs[mask_daytime]).quantile(0.99)
                p_cs = ghi_cs * max_ratio
            else:
                p_cs = ghi_cs
        
        return p_cs.clip(lower=0)
    
    def calculate_kt(
        self,
        times: pd.DatetimeIndex,
        actual_power: pd.Series,
        epsilon: float = 1.0,
        kt_max: float = 1.5,
        measured_ghi=None,
        fit_mask: Optional[np.ndarray] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calcule l'indice de clarté kt = P_actual / P_clearsky.

        Args:
            times: Index temporel
            actual_power: Puissance réelle
            epsilon: Valeur minimale pour éviter division par zéro
            kt_max: Valeur maximale de kt (pour éviter aberrations)
            measured_ghi: GHI mesuré (détection des périodes de ciel clair)
            fit_mask: lignes d'entraînement utilisées pour ajuster P_cs

        Returns:
            (kt_series, p_clearsky_series)
        """
        p_cs = self.calculate_clear_sky_power(times, actual_power, measured_ghi, fit_mask)
        p_cs = pd.Series(np.asarray(p_cs, dtype=float), index=actual_power.index)

        # Calcul de kt avec régularisation
        kt = actual_power / (p_cs + epsilon)

        # Clipper les valeurs aberrantes
        kt = kt.clip(lower=0, upper=kt_max)

        # kt = 0 pendant la nuit (P_cs < seuil)
        night_mask = p_cs < epsilon
        kt[night_mask] = 0

        day = ~night_mask
        clipped = float(np.isclose(kt[day], kt_max).mean()) if day.any() else 0.0
        (logger.warning if clipped > 0.02 else logger.info)(
            f"   k_t at the {kt_max} clip: {clipped:.1%} of daytime samples")

        return kt, p_cs

    def add_physics_features(
        self,
        df: pd.DataFrame,
        power_column: str = "Active_Power",
        epsilon: float = 1.0,
        kt_max: float = 1.5,
        ghi_column: str = "GHI",
        fit_fraction: Optional[float] = None,
        fit_mask: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Ajoute les features physiques au DataFrame.
        
        Features ajoutées:
        - zenith_angle: Angle zénithal du soleil
        - clear_sky_ghi: GHI théorique de ciel clair
        - clear_sky_power: Puissance théorique
        - kt: Indice de clarté (cible normalisée)
        
        IMPORTANT: S'assure que le fuseau horaire est correct pour
        le calcul astronomique.
        """
        logger.info("🔬 Ajout des features physiques...")
        
        df = df.copy()
        times = df.index
        
        # Vérifier le fuseau horaire
        if times.tz is None:
            logger.info(f"   ⚠️ Localisation de l'index en {self.timezone}")
            times = times.tz_localize(self.timezone)
            df.index = times
        
        # Position solaire
        solar_pos = self.calculate_solar_position(times)
        df['zenith_angle'] = solar_pos['zenith'].values
        
        # GHI de ciel clair
        df['clear_sky_ghi'] = self.calculate_clear_sky_ghi(times).values
        
        # Puissance et kt
        if power_column in df.columns:
            actual_power = df[power_column]
            measured_ghi = df[ghi_column] if ghi_column in df.columns else None
            # P_cs is fitted on the leading (training) rows only: fit_fraction = train_ratio.
            # An explicit fit_mask (training rows) takes precedence, e.g. with interleaved validation.
            if fit_mask is None:
                n_fit = int(len(df) * fit_fraction) if fit_fraction else len(df)
                fit_mask = np.arange(len(df)) < n_fit
            kt, p_cs = self.calculate_kt(times, actual_power, epsilon, kt_max,
                                         measured_ghi=measured_ghi, fit_mask=fit_mask)
            df['clear_sky_power'] = p_cs.values
            df['kt'] = kt.values
            
            # Stats
            daytime_mask = df['clear_sky_ghi'] > 10
            kt_mean = df.loc[daytime_mask, 'kt'].mean()
            logger.info(f"   ✅ kt moyen (diurne): {kt_mean:.3f}")
        
        logger.info(f"   ✅ Features ajoutées: zenith_angle, clear_sky_ghi, clear_sky_power, kt")
        
        return df


def load_config_and_create_calculator(config_path: str) -> ClearSkyCalculator:
    """
    Charge la configuration et crée un ClearSkyCalculator.
    
    Usage:
        calculator = load_config_and_create_calculator("configs/data_config_yulara_neighbours.yaml")
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    loc = config['location']
    preproc = config.get('preprocessing', {})
    
    return ClearSkyCalculator(
        latitude=loc['latitude'],
        longitude=loc['longitude'],
        altitude=loc.get('altitude', 0),
        timezone=loc['timezone'],
        model=preproc.get('clear_sky_model', 'ineichen'),
        array_tilt=loc.get('array_tilt'),
        array_azimuth=loc.get('array_azimuth'),
        power_model=preproc.get('clear_sky_power_model', 'poa_clear')
    )


# Test du module
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test avec Yulara PV Power Plant, Uluru, Australie
    calc = ClearSkyCalculator(
        latitude=-25.24,
        longitude=131.04,
        altitude=492,
        timezone="Australia/Darwin"
    )
    
    # Créer un index de test
    times = pd.date_range(
        start='2021-06-01',
        periods=288,
        freq='5min',
        tz="Australia/Darwin"
    )
    
    # Calculer GHI
    ghi_cs = calc.calculate_clear_sky_ghi(times)
    print(f"\nGHI Clear Sky (max): {ghi_cs.max():.1f} W/m²")
    print(f"GHI Clear Sky (mean diurne): {ghi_cs[ghi_cs > 10].mean():.1f} W/m²")
