"""
RoutingLogger — Logging par échantillon des décisions de routage AHSE
=====================================================================
Produit pour chaque échantillon du test set une ligne de log :
    sample_idx | sigma_kt | regime | selected_model | y_true_mean | y_pred_mean | squared_error

Usage typique :
    from src.utils.routing_logger import RoutingLogger

    logger = RoutingLogger(output_path=Path("results/yulara_neighbours/tables/routing_log_1h.csv"),
                           tau=0.15,
                           model_stable="LightGBM",
                           model_variable="PatchTST")

    # Dans la boucle de prédiction AHSE :
    logger.log_batch(variances, stable_mask, y_true, y_pred)
    logger.save()

Figures produites via plot() :
    1. Distribution σ(kₜ) avec τ surimposé (violin + KDE)
    2. Proportion stable/variable par heure de la journée
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RoutingLogger:
    """
    Enregistre les décisions de routage du GatingNetwork échantillon par échantillon.

    Attributs stockés par appel à log_batch() :
        sample_idx    : indice global de l'échantillon dans le test set
        sigma_kt      : écart-type local de kₜ (variance de clarté)
        regime        : 'stable' si sigma_kt < tau, 'variable' sinon
        selected_model: modèle utilisé pour cet échantillon
        y_true_mean   : moyenne de y_true sur les 96 pas de l'horizon
        y_pred_mean   : moyenne de y_pred sur les 96 pas de l'horizon
        squared_error : MSE moyen sur les 96 pas (pour analyse d'erreur par régime)
    """

    def __init__(
        self,
        output_path: Path,
        tau: float = 0.15,
        model_stable: str = "LightGBM",
        model_variable: str = "PatchTST",
    ) -> None:
        self.output_path = Path(output_path)
        self.tau = tau
        self.model_stable = model_stable
        self.model_variable = model_variable
        self._records: list[dict] = []
        self._offset: int = 0  # décalage d'index si appelé en plusieurs batches

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_batch(
        self,
        variances: np.ndarray,
        stable_mask: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> None:
        """
        Enregistre les décisions pour un batch d'échantillons.

        Args:
            variances   : sigma_kt par échantillon, shape (n_samples,)
            stable_mask : booléen par échantillon (True = stable), shape (n_samples,)
            y_true      : valeurs réelles, shape (n_samples, horizon) ou None
            y_pred      : prédictions AHSE, shape (n_samples, horizon) ou None
            timestamps  : timestamps optionnels, shape (n_samples,)
        """
        n = len(variances)
        for i in range(n):
            rec: dict = {
                "sample_idx":     self._offset + i,
                "sigma_kt":       float(variances[i]),
                "regime":         "stable" if stable_mask[i] else "variable",
                "selected_model": self.model_stable if stable_mask[i] else self.model_variable,
            }

            if y_true is not None:
                rec["y_true_mean"] = float(np.mean(y_true[i]))
            if y_pred is not None:
                rec["y_pred_mean"] = float(np.mean(y_pred[i]))
            if y_true is not None and y_pred is not None:
                rec["squared_error"] = float(np.mean((y_true[i] - y_pred[i]) ** 2))
            if timestamps is not None:
                rec["timestamp"] = timestamps[i]

            self._records.append(rec)

        self._offset += n
        logger.debug("RoutingLogger : %d échantillons enregistrés (total=%d)", n, self._offset)

    def reset(self) -> None:
        """Remet le logger à zéro (utile entre deux runs de sensibilité τ)."""
        self._records = []
        self._offset = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> pd.DataFrame:
        """Sauvegarde les logs en CSV et retourne le DataFrame."""
        if not self._records:
            logger.warning("RoutingLogger.save() : aucun enregistrement à sauvegarder.")
            return pd.DataFrame()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._records)
        df.to_csv(self.output_path, index=False)
        logger.info("RoutingLogger : %d lignes sauvegardées → %s", len(df), self.output_path)
        return df

    def load(self) -> pd.DataFrame:
        """Charge un log CSV existant."""
        return pd.read_csv(self.output_path)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def plot(
        self,
        figures_dir: Optional[Path] = None,
        show: bool = False,
    ) -> None:
        """
        Produit deux figures de la limitation #2 :
          Fig A — Distribution σ(kₜ) avec τ surimposé (KDE + ligne verticale)
          Fig B — Proportion stable/variable par heure de la journée

        Args:
            figures_dir : dossier de sortie (None = même dossier que le CSV)
            show        : afficher les figures interactivement
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            logger.error("matplotlib manquant — impossible de générer les figures.")
            return

        if not self._records:
            logger.warning("RoutingLogger.plot() : aucune donnée, chargement depuis %s", self.output_path)
            try:
                df = self.load()
            except FileNotFoundError:
                logger.error("Fichier CSV introuvable : %s", self.output_path)
                return
        else:
            df = pd.DataFrame(self._records)

        if figures_dir is None:
            figures_dir = self.output_path.parent
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Palette colorblind-friendly
        COLOR_STABLE   = "#2196F3"   # bleu
        COLOR_VARIABLE = "#FF5722"   # orange
        COLOR_TAU      = "#4CAF50"   # vert

        # ── Figure A : Distribution σ(kₜ) ──────────────────────────────
        fig_a, ax_a = plt.subplots(figsize=(7, 4))

        sigma_vals = df["sigma_kt"].values
        ax_a.hist(sigma_vals, bins=80, density=True, alpha=0.55,
                  color="#607D8B", label="σ(kₜ) distribution")

        # KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(sigma_vals, bw_method="scott")
            x_grid = np.linspace(sigma_vals.min(), sigma_vals.max(), 300)
            ax_a.plot(x_grid, kde(x_grid), color="#212121", lw=1.5, label="KDE")
        except ImportError:
            pass

        ax_a.axvline(self.tau, color=COLOR_TAU, lw=2, ls="--",
                     label=f"τ = {self.tau}")

        pct_stable = (sigma_vals < self.tau).mean() * 100
        ax_a.text(self.tau + 0.005, ax_a.get_ylim()[1] * 0.85,
                  f"{pct_stable:.1f}% stable\n{100-pct_stable:.1f}% variable",
                  fontsize=8, color="#212121")

        ax_a.set_xlabel("σ(kₜ) — écart-type local de l'indice de clarté", fontsize=10)
        ax_a.set_ylabel("Densité", fontsize=10)
        ax_a.set_title("Distribution de la variabilité du ciel", fontsize=11)
        ax_a.legend(fontsize=9)
        ax_a.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        fig_a.tight_layout()

        path_a = figures_dir / "routing_sigma_kt_distribution.pdf"
        fig_a.savefig(path_a, dpi=300, bbox_inches="tight")
        fig_a.savefig(path_a.with_suffix(".png"), dpi=300, bbox_inches="tight")
        logger.info("Figure A sauvegardée : %s", path_a)

        # ── Figure B : Proportion stable/variable par heure ─────────────
        if "timestamp" in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        elif "sample_idx" in df.columns:
            # Approximation si pas de timestamp : supposer 15-min resolution, début à 00:00
            df["hour"] = ((df["sample_idx"] % 96) * 15 // 60)

        hourly = df.groupby("hour")["regime"].value_counts(normalize=True).unstack(fill_value=0)
        if "stable" not in hourly.columns:
            hourly["stable"] = 0.0
        if "variable" not in hourly.columns:
            hourly["variable"] = 0.0

        fig_b, ax_b = plt.subplots(figsize=(9, 4))
        hours = hourly.index
        ax_b.bar(hours, hourly["stable"] * 100, label="Stable (LightGBM)",
                 color=COLOR_STABLE, alpha=0.8)
        ax_b.bar(hours, hourly["variable"] * 100, bottom=hourly["stable"] * 100,
                 label="Variable (PatchTST)", color=COLOR_VARIABLE, alpha=0.8)

        ax_b.set_xlabel("Heure de la journée (UTC+9:30)", fontsize=10)
        ax_b.set_ylabel("Proportion (%)", fontsize=10)
        ax_b.set_title(f"Routage AHSE par heure — τ = {self.tau}", fontsize=11)
        ax_b.set_xticks(range(0, 24, 2))
        ax_b.set_ylim(0, 100)
        ax_b.legend(fontsize=9)
        fig_b.tight_layout()

        path_b = figures_dir / "routing_hourly_proportion.pdf"
        fig_b.savefig(path_b, dpi=300, bbox_inches="tight")
        fig_b.savefig(path_b.with_suffix(".png"), dpi=300, bbox_inches="tight")
        logger.info("Figure B sauvegardée : %s", path_b)

        if show:
            plt.show()
        plt.close("all")

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Retourne un résumé statistique des décisions de routage."""
        if not self._records:
            return {}
        df = pd.DataFrame(self._records)
        n = len(df)
        n_stable = (df["regime"] == "stable").sum()
        return {
            "n_samples":       n,
            "n_stable":        int(n_stable),
            "n_variable":      int(n - n_stable),
            "pct_stable":      float(n_stable / n * 100),
            "sigma_kt_mean":   float(df["sigma_kt"].mean()),
            "sigma_kt_median": float(df["sigma_kt"].median()),
            "tau":             self.tau,
        }
