"""Detección de deriva de datos y de predicciones mediante el índice de estabilidad poblacional (PSI)."""
from __future__ import annotations

import numpy as np

UMBRAL_ALERTA = 0.2  # convención usual: PSI > 0,2 indica un cambio significativo


def psi(referencia, actual, n_bins: int = 10, eps: float = 1e-4) -> float:
    referencia, actual = np.asarray(referencia, float), np.asarray(actual, float)
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(referencia, cortes)[0] / len(referencia) + eps
    q = np.histogram(actual, cortes)[0] / len(actual) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorico(referencia, actual, categorias=range(5), eps: float = 1e-4) -> float:
    p = np.array([np.mean(np.asarray(referencia) == c) for c in categorias]) + eps
    q = np.array([np.mean(np.asarray(actual) == c) for c in categorias]) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def evaluar_deriva(estad_ref: dict[str, list], estad_act: dict[str, list]) -> dict[str, dict]:
    """Calcula el PSI por estadística de imagen y marca alerta cuando supera el umbral."""
    return {k: {"psi": (v := psi(estad_ref[k], estad_act[k])), "alerta": v > UMBRAL_ALERTA} for k in estad_ref}
