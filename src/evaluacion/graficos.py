"""Gráficos estáticos de resultados sobre simulación reproducible."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluacion.metricas import captura_top_20, curva_valor_capturado, tabla_lift

PINO = "#276653"
DORADO = "#c98c22"
AZUL = "#587080"
COLORES = {"logistica": PINO, "gradient_boosting": DORADO, "combinacion": AZUL}
NOMBRES = {"logistica": "Logística", "gradient_boosting": "Gradient boosting", "combinacion": "Combinación"}


def _estilo() -> None:
    plt.rcParams.update(
        {"figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
         "axes.labelcolor": "#34423c", "text.color": "#263a32", "font.size": 10,
         "axes.edgecolor": "#b9c6bf", "axes.titleweight": "bold", "grid.alpha": 0.17}
    )


def guardar_figuras(
    y: np.ndarray,
    predicciones: dict[str, np.ndarray],
    saldos: np.ndarray,
    destino: Path,
    *,
    baselines: dict[str, np.ndarray] | None = None,
) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    _estilo()
    # Discriminación: ROC y precisión-recall (la base de referencia importa).
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.6), layout="constrained")
    for nombre, prob in predicciones.items():
        fpr, tpr, _ = roc_curve(y, prob)
        precisiones, recalls, _ = precision_recall_curve(y, prob)
        axes[0].plot(fpr, tpr, color=COLORES[nombre], lw=2, label=NOMBRES[nombre])
        axes[1].plot(recalls, precisiones, color=COLORES[nombre], lw=2, label=NOMBRES[nombre])
    axes[0].plot([0, 1], [0, 1], "--", color="#a5aca8", lw=1)
    axes[1].axhline(float(np.mean(y)), ls="--", color="#a5aca8", label="Prevalencia")
    for ax, titulo, x, ylabel in [(axes[0], "ROC · test por cosecha", "Tasa de falsos positivos", "Sensibilidad"),
                                  (axes[1], "Precisión–recall · test", "Recall", "Precisión")]:
        ax.set(title=titulo, xlabel=x, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
    fig.savefig(destino / "discriminacion.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="#a5aca8", label="Calibración ideal")
    for nombre, prob in predicciones.items():
        observada, estimada = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        ax.plot(estimada, observada, marker="o", markersize=4, lw=2, color=COLORES[nombre], label=NOMBRES[nombre])
    ax.set(title="Calibración · probabilidades vs. frecuencia observada", xlabel="Probabilidad estimada", ylabel="Frecuencia de curación", xlim=(0, 1), ylim=(0, 1))
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.savefig(destino / "calibracion.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.9), layout="constrained")
    for nombre, prob in predicciones.items():
        agrupado = tabla_lift(y, prob, saldos, nombre)
        ax.plot(agrupado["decil"], agrupado["lift"], "o-", lw=2, color=COLORES[nombre], label=NOMBRES[nombre])
    ax.axhline(1.0, linestyle="--", color="#a5aca8", label="Selección aleatoria")
    ax.set(title="Lift de curación por decil de probabilidad", xlabel="Decil (1 = mayor probabilidad)", ylabel="Lift frente a prevalencia", xticks=np.arange(1, 11))
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.savefig(destino / "lift.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.1), layout="constrained")
    ax.plot([0, 100], [0, 100], "--", color="#a5aca8", label="Priorización aleatoria (referencia)")
    for nombre, prob in predicciones.items():
        curva = curva_valor_capturado(y, prob, saldos)
        ax.plot(curva["porcentaje_cuentas_priorizadas"], curva["saldo_curado_capturado_pct"], "o-", lw=2, color=COLORES[nombre], label=NOMBRES[nombre])
    ax.set(title="Concentración del saldo en cuentas que curan · test sintético", xlabel="% de cuentas priorizadas (orden p × saldo)", ylabel="% de saldo de cuentas que curaron (no efectivo)", xlim=(0, 100), ylim=(0, 103))
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.savefig(destino / "valor_capturado.png", dpi=170)
    plt.close(fig)

    if baselines is not None:
        referencia = max(
            captura_top_20(y, np.zeros(len(y)), saldos, prioridad=orden)
            for orden in baselines.values()
        )
        etiquetas = ["Saldo descendente", "DPD 30→60 (desempate: saldo)"] + [
            NOMBRES[n] for n in predicciones
        ]
        valores = [
            captura_top_20(y, np.zeros(len(y)), saldos, prioridad=p)
            for p in baselines.values()
        ] + [captura_top_20(y, p, saldos) for p in predicciones.values()]
        colores = ["#a2aba5", "#6d8075"] + [COLORES[n] for n in predicciones]
        fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
        barras = ax.barh(etiquetas, valores, color=colores, height=0.62)
        ax.invert_yaxis()
        ax.axvline(referencia, ls="--", lw=1.5, color="#485a51", label="Mejor regla simple")
        for idx, (barra, valor) in enumerate(zip(barras, valores)):
            sufijo = "" if idx < 2 else f"  (Δ {valor - referencia:+.2f} pp)"
            ax.text(valor + 0.3, barra.get_y() + barra.get_height() / 2, f"{valor:.2f}%{sufijo}", va="center", fontsize=9)
        ax.set(
            title="Saldo de cuentas curadas en top 20% · reglas vs. ML",
            xlabel="% de saldo que posteriormente curó · simulación, no efectivo cobrado",
            xlim=(0, min(100, max(valores) + 24)),
        )
        ax.grid(axis="x", alpha=0.17)
        ax.legend(fontsize=9, loc="lower right")
        fig.savefig(destino / "valor_incremental.png", dpi=170)
        plt.close(fig)
