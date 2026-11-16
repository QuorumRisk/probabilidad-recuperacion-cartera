"""Construcción de target mediante meses futuros, separados de las features."""

from __future__ import annotations

import pandas as pd

HORIZONTE = 3
EDAD_MIN = 4
EDAD_MAX = 9


def definir_observaciones(panel: pd.DataFrame, horizonte: int = HORIZONTE) -> pd.DataFrame:
    """Una observación por préstamo: primera mora 30/60 entre edades 4–9.

    La historia debe contener t−3, t−2, t−1 (filas reales, consecutivas
    y con atributos observados). `label_cura_3m` exige t+1..t+3 completos,
    consecutivos y con DPD no nulo. Lo desconocido nunca se etiqueta cero.
    """
    if horizonte != HORIZONTE:
        raise ValueError("Este demostrador documenta un horizonte fijo de tres meses.")
    if panel.duplicated(["loan_id", "mes_edad"]).any():
        raise ValueError("El panel contiene filas duplicadas por crédito y mes.")

    ordenado = panel.sort_values(["loan_id", "mes_edad"], kind="stable").copy()
    por_credito = ordenado.groupby("loan_id", sort=False)
    campos_historicos = [
        "dias_mora", "pago_realizado", "cuota_programada", "saldo_actual", "fecha_reporte"
    ]
    historia_valida = pd.Series(True, index=ordenado.index)
    for rezago in range(1, EDAD_MIN):
        # `shift()` sin verificar mes real aceptaría enero y marzo como
        # si fueran dos meses consecutivos. Exigir edad t−k cierra ese hueco.
        historia_valida &= por_credito["mes_edad"].shift(rezago).eq(
            ordenado["mes_edad"] - rezago
        )
        for campo in campos_historicos:
            historia_valida &= por_credito[campo].shift(rezago).notna()

    mes_valido = ordenado[
        ["pago_realizado", "cuota_programada", "saldo_vencido_cierre", "fecha_reporte"]
    ].notna().all(axis=1)
    candidatos = ordenado.loc[
        ordenado["dias_mora"].isin((30, 60))
        & ordenado["mes_edad"].between(EDAD_MIN, EDAD_MAX)
        & (ordenado["saldo_actual"] > 0)
        & mes_valido
        & historia_valida
    ].copy()
    decisiones = candidatos.drop_duplicates("loan_id", keep="first").copy()
    if decisiones.empty:
        raise ValueError("No se generaron decisiones elegibles en mora 30–60 días.")

    futuros = ordenado[["loan_id", "mes_edad", "dias_mora"]].merge(
        decisiones[["loan_id", "mes_edad"]].rename(columns={"mes_edad": "mes_decision"}),
        on="loan_id",
        validate="many_to_one",
    )
    futuros = futuros.loc[
        futuros["mes_edad"].gt(futuros["mes_decision"])
        & futuros["mes_edad"].le(futuros["mes_decision"] + horizonte)
    ].copy()
    futuros["al_dia"] = futuros["dias_mora"].eq(0)
    futuros["resultado_conocido"] = futuros["dias_mora"].notna()
    etiquetas = futuros.groupby("loan_id").agg(
        meses_observados=("mes_edad", "size"),
        primer_mes=("mes_edad", "min"),
        ultimo_mes=("mes_edad", "max"),
        meses_con_dpd_valido=("resultado_conocido", "sum"),
        label_cura_3m=("al_dia", "max"),
    )
    mes_partida = decisiones.set_index("loan_id")["mes_edad"]
    etiquetas = etiquetas.join(mes_partida.rename("mes_partida"))
    etiquetas = etiquetas.loc[
        etiquetas["meses_observados"].eq(horizonte)
        & etiquetas["meses_con_dpd_valido"].eq(horizonte)
        & etiquetas["primer_mes"].eq(etiquetas["mes_partida"] + 1)
        & etiquetas["ultimo_mes"].eq(etiquetas["mes_partida"] + horizonte)
    ]
    decisiones = decisiones.merge(
        etiquetas[["label_cura_3m"]], left_on="loan_id", right_index=True, how="inner", validate="one_to_one"
    )
    decisiones["label_cura_3m"] = decisiones["label_cura_3m"].astype(int)
    decisiones = decisiones.rename(columns={"saldo_actual": "saldo_decision", "fecha_reporte": "fecha_decision"})
    decisiones["saldo_cura_observado"] = decisiones["saldo_decision"] * decisiones["label_cura_3m"]
    return decisiones.reset_index(drop=True)
