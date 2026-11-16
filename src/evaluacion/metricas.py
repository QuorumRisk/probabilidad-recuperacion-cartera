"""Evaluación fuera de muestra de discriminación, calibración y priorización."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calcular_metricas(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Reporta AUC, KS, PR-AUC y métricas de probabilidad/umbral ilustrativo."""
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Se necesitan ambas clases para calcular AUC y KS.")
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Las predicciones deben ser probabilidades válidas.")
    fpr, tpr, _ = roc_curve(y, prob)
    estimadas = (prob >= 0.5).astype(int)
    return {
        "auc_roc": round(float(roc_auc_score(y, prob)), 6),
        "ks": round(float(np.max(tpr - fpr)), 6),
        "pr_auc_average_precision": round(float(average_precision_score(y, prob)), 6),
        "brier": round(float(brier_score_loss(y, prob)), 6),
        "log_loss": round(float(log_loss(y, prob, labels=[0, 1])), 6),
        "precision_umbral_050": round(float(precision_score(y, estimadas, zero_division=0)), 6),
        "recall_umbral_050": round(float(recall_score(y, estimadas, zero_division=0)), 6),
        "prevalencia_cura": round(float(y.mean()), 6),
    }


def tabla_lift(y: np.ndarray, prob: np.ndarray, saldos: np.ndarray, modelo: str) -> pd.DataFrame:
    """Deciles por probabilidad de cura, sin usar la etiqueta para ordenar."""
    tabla = pd.DataFrame({"y": y, "prob": prob, "saldo": saldos})
    tabla = tabla.sort_values("prob", ascending=False, kind="stable").reset_index(drop=True)
    tabla["decil"] = (np.arange(len(tabla)) * 10 // len(tabla) + 1).clip(1, 10)
    tabla["saldo_curado"] = tabla["y"] * tabla["saldo"]
    agrupada = tabla.groupby("decil", sort=True).agg(
        cuentas=("y", "size"),
        cuentas_curadas=("y", "sum"),
        tasa_cura=("y", "mean"),
        saldo_curado=("saldo_curado", "sum"),
    ).reset_index()
    prevalencia = tabla["y"].mean()
    agrupada["lift"] = agrupada["tasa_cura"] / prevalencia if prevalencia else 0.0
    agrupada["cuentas_curadas_acumuladas_pct"] = (
        agrupada["cuentas_curadas"].cumsum() / max(1, tabla["y"].sum()) * 100
    )
    agrupada.insert(0, "modelo", modelo)
    return agrupada.round(4)


def curva_valor_capturado(y: np.ndarray, prob: np.ndarray, saldos: np.ndarray) -> pd.DataFrame:
    """Captura del *saldo que curó*, NO del dinero efectivamente cobrado.

    Ordenamos cuentas por probabilidad de cura × saldo observado al decidir.
    La etiqueta futura se usa solo para medir el resultado, nunca para ordenar.
    """
    tabla = pd.DataFrame({"y": y, "prob": prob, "saldo": saldos})
    tabla["prioridad"] = tabla["prob"] * tabla["saldo"]
    tabla["saldo_curado"] = tabla["saldo"] * tabla["y"]
    tabla = tabla.sort_values("prioridad", ascending=False, kind="stable").reset_index(drop=True)
    total = max(1e-12, float(tabla["saldo_curado"].sum()))
    acumulado = tabla["saldo_curado"].cumsum().to_numpy()
    n = len(tabla)
    indices = [int(np.ceil(n * fraccion / 10)) for fraccion in range(11)]
    porcentajes = [0.0] + [float(acumulado[k - 1] / total * 100) for k in indices[1:]]
    return pd.DataFrame(
        {"porcentaje_cuentas_priorizadas": np.arange(0, 101, 10), "saldo_curado_capturado_pct": porcentajes}
    )


def captura_top_20(y: np.ndarray, prob: np.ndarray, saldos: np.ndarray) -> float:
    curva = curva_valor_capturado(y, prob, saldos)
    return round(float(curva.loc[curva["porcentaje_cuentas_priorizadas"].eq(20), "saldo_curado_capturado_pct"].iloc[0]), 6)
