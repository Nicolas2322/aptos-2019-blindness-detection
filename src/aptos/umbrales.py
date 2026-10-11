"""Conversión de la salida continua a grados y ajuste de umbrales para maximizar el QWK."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

UMBRALES_INICIALES = (0.5, 1.5, 2.5, 3.5)


def qwk(y_real, y_pred) -> float:
    """Kappa de Cohen con ponderación cuadrática, métrica oficial de la competencia."""
    return float(cohen_kappa_score(y_real, y_pred, weights="quadratic"))


def a_grados(valores, umbrales=UMBRALES_INICIALES) -> np.ndarray:
    umbrales = np.sort(np.asarray(umbrales, dtype=float))
    return np.searchsorted(umbrales, np.clip(np.asarray(valores, dtype=float), 0.0, 4.0)).astype(int)


class OptimizadorUmbrales:
    """Ajusta los cuatro umbrales con Nelder-Mead sobre las predicciones fuera de partición."""

    def __init__(self, iniciales=UMBRALES_INICIALES):
        self.umbrales_ = np.asarray(iniciales, dtype=float)

    def fit(self, valores, y_real):
        objetivo = lambda u: -qwk(y_real, a_grados(valores, u))
        res = minimize(objetivo, self.umbrales_, method="Nelder-Mead", options={"xatol": 1e-3, "maxiter": 400})
        if -res.fun >= qwk(y_real, a_grados(valores, self.umbrales_)):
            self.umbrales_ = np.sort(res.x)
        return self

    def predict(self, valores) -> np.ndarray:
        return a_grados(valores, self.umbrales_)
