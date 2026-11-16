"""Ingeniería temporal: solo hechos disponibles al cierre del mes de decisión.

La tabla de predictores se construye con una *lista positiva* de variables.
No se arrastran columnas futuras ni columnas objetivo accidentalmente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAS = ["grado_riesgo", "proposito"]
NUMERICAS = [
    "mes_edad",
    "ingreso_mensual",
    "dti_originacion",
    "principal_original",
    "plazo_meses",
    "tasa_anual",
    "cuota_sobre_ingreso",
    "saldo_decision",
    "ratio_saldo_original",
    "saldo_vencido_decision",
    "cuotas_vencidas_equivalentes",
    "dias_mora_decision",
    "pago_ratio_ultimo_mes",
    "pago_ratio_media_3m",
    "pago_ratio_volatilidad_3m",
    "pago_ratio_tendencia_3m",
    "meses_en_mora_3m",
    "meses_en_mora_6m",
    "peor_mora_6m",
    "dpd_tendencia_3m",
    "recencia_ultimo_al_dia_6m",
    "ratio_pago_vs_cuota_6m",
]
COLUMNAS_MODELO = NUMERICAS + CATEGORICAS
METADATOS = ["loan_id", "cosecha", "fecha_decision", "label_cura_3m", "saldo_cura_observado"]


def construir_variables(panel: pd.DataFrame, decisiones: pd.DataFrame) -> pd.DataFrame:
    """Devuelve observaciones con predictores históricos y etiqueta aislada.

    El join utiliza `mes_edad <= mes_decision` **antes** de cualquier agregación.
    Prueba anti-fuga: modificar meses posteriores no altera este resultado.
    """
    if decisiones["loan_id"].duplicated().any():
        raise ValueError("Debe existir una única decisión por préstamo.")
    decision_ref = decisiones[["loan_id", "mes_edad"]].rename(columns={"mes_edad": "mes_decision"})
    historia = panel.merge(decision_ref, on="loan_id", how="inner", validate="many_to_one")
    historia = historia.loc[historia["mes_edad"].le(historia["mes_decision"])].copy()
    historia = historia.sort_values(["loan_id", "mes_edad"])
    historia["pago_ratio"] = np.clip(
        historia["pago_realizado"] / historia["cuota_programada"].clip(lower=0.01), 0, 3
    )
    historia["es_mora"] = historia["dias_mora"].gt(0).astype(int)
    historia["ratio_duracion"] = historia["pago_realizado"] / historia["cuota_programada"].clip(lower=0.01)

    h3 = historia.loc[historia["mes_edad"].gt(historia["mes_decision"] - 3)]
    h6 = historia.loc[historia["mes_edad"].gt(historia["mes_decision"] - 6)]

    p3 = h3.groupby("loan_id").agg(
        pago_ratio_media_3m=("pago_ratio", "mean"),
        pago_ratio_volatilidad_3m=("pago_ratio", "std"),
        meses_en_mora_3m=("es_mora", "sum"),
    )
    p3["pago_ratio_tendencia_3m"] = h3.groupby("loan_id")["pago_ratio"].agg(
        lambda serie: float(serie.iloc[-1] - serie.iloc[0])
    )
    p3["dpd_tendencia_3m"] = h3.groupby("loan_id")["dias_mora"].agg(
        lambda serie: float(serie.iloc[-1] - serie.iloc[0])
    )

    p6 = h6.groupby("loan_id").agg(
        meses_en_mora_6m=("es_mora", "sum"),
        peor_mora_6m=("dias_mora", "max"),
        ratio_pago_vs_cuota_6m=("ratio_duracion", "mean"),
    )
    ultimo_al_dia = h6.loc[h6["dias_mora"].eq(0)].groupby("loan_id")["mes_edad"].max()
    p6["ultimo_mes_al_dia"] = ultimo_al_dia

    columnas_decision = [
        "loan_id", "cosecha", "fecha_decision", "label_cura_3m", "saldo_cura_observado",
        "mes_edad", "ingreso_anual", "dti_originacion", "principal_original", "plazo_meses",
        "tasa_anual", "cuota_programada", "saldo_decision", "saldo_vencido_cierre",
        "dias_mora", "grado_riesgo",
        "proposito", "pago_realizado",
    ]
    tabla = decisiones[columnas_decision].set_index("loan_id").join(p3).join(p6)
    tabla["ingreso_mensual"] = tabla["ingreso_anual"] / 12.0
    tabla["cuota_sobre_ingreso"] = tabla["cuota_programada"] / tabla["ingreso_mensual"].clip(lower=0.01)
    tabla["ratio_saldo_original"] = tabla["saldo_decision"] / tabla["principal_original"].clip(lower=0.01)
    tabla["saldo_vencido_decision"] = tabla["saldo_vencido_cierre"]
    tabla["cuotas_vencidas_equivalentes"] = tabla["saldo_vencido_cierre"] / tabla["cuota_programada"].clip(lower=0.01)
    tabla["dias_mora_decision"] = tabla["dias_mora"]
    tabla["pago_ratio_ultimo_mes"] = tabla["pago_realizado"] / tabla["cuota_programada"].clip(lower=0.01)
    tabla["recencia_ultimo_al_dia_6m"] = (
        tabla["mes_edad"] - tabla["ultimo_mes_al_dia"]
    ).fillna(7.0)
    tabla["pago_ratio_volatilidad_3m"] = tabla["pago_ratio_volatilidad_3m"].fillna(0.0)

    columnas = ["loan_id"] + METADATOS[1:] + COLUMNAS_MODELO
    resultado = tabla.reset_index()[columnas].copy()
    if resultado[COLUMNAS_MODELO].isna().any().any():
        raise ValueError("Existen predictores sin definir después de construir la historia.")
    return resultado
