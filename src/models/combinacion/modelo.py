"""Combinación por promedio ponderado; peso seleccionado en validación, no test."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss

PESOS_CANDIDATOS = (0.25, 0.50, 0.75)


def combinar(p_logistica: np.ndarray, p_boosting: np.ndarray, peso_logistica: float) -> np.ndarray:
    if not 0.0 <= peso_logistica <= 1.0:
        raise ValueError("El peso debe estar entre cero y uno.")
    return peso_logistica * np.asarray(p_logistica) + (1.0 - peso_logistica) * np.asarray(p_boosting)


def seleccionar_peso(y_validacion: np.ndarray, p_logistica: np.ndarray, p_boosting: np.ndarray) -> tuple[float, dict[str, float]]:
    """Minimiza Brier únicamente en cosecha 2021; nunca inspecciona 2022."""
    briers = {str(p): float(brier_score_loss(y_validacion, combinar(p_logistica, p_boosting, p))) for p in PESOS_CANDIDATOS}
    mejor = min(PESOS_CANDIDATOS, key=lambda p: (briers[str(p)], abs(p - 0.5)))
    return mejor, briers
