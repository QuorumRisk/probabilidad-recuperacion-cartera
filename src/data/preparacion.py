"""Construcción de target mediante meses futuros, separados de las features."""

from __future__ import annotations

import pandas as pd

HORIZONTE = 3
EDAD_MIN = 4
EDAD_MAX = 9


def definir_observaciones(panel: pd.DataFrame, horizonte: int = HORIZONTE) -> pd.DataFrame:
    """Una observación por préstamo: primera mora 30/60 entre edades 4–9.

    `label_cura_3m` requiere exactamente 3 periodos posteriores y se construye
    exclusivamente para evaluación; nunca se incorpora a las variables.
    """
    if horizonte != HORIZONTE:
        raise ValueError("Este demostrador documenta un horizonte fijo de tres meses.")
    if panel.duplicated(["loan_id", "mes_edad"]).any():
        raise ValueError("El panel contiene filas duplicadas por crédito y mes.")

    candidatos = panel.loc[
        panel["dias_mora"].isin((30, 60))
        & panel["mes_edad"].between(EDAD_MIN, EDAD_MAX)
        & (panel["saldo_actual"] > 0)
    ].sort_values(["loan_id", "mes_edad"])
    decisiones = candidatos.drop_duplicates("loan_id", keep="first").copy()
    if decisiones.empty:
        raise ValueError("No se generaron decisiones elegibles en mora 30–60 días.")

    futuros = panel[["loan_id", "mes_edad", "dias_mora"]].merge(
        decisiones[["loan_id", "mes_edad"]].rename(columns={"mes_edad": "mes_decision"}),
        on="loan_id",
        validate="many_to_one",
    )
    futuros = futuros.loc[
        futuros["mes_edad"].gt(futuros["mes_decision"])
        & futuros["mes_edad"].le(futuros["mes_decision"] + horizonte)
    ].copy()
    futuros["al_dia"] = futuros["dias_mora"].eq(0)
    etiquetas = futuros.groupby("loan_id").agg(
        meses_observados=("mes_edad", "size"),
        label_cura_3m=("al_dia", "max"),
    )
    etiquetas = etiquetas.loc[etiquetas["meses_observados"].eq(horizonte)]
    decisiones = decisiones.merge(
        etiquetas[["label_cura_3m"]], left_on="loan_id", right_index=True, how="inner", validate="one_to_one"
    )
    decisiones["label_cura_3m"] = decisiones["label_cura_3m"].astype(int)
    decisiones = decisiones.rename(columns={"saldo_actual": "saldo_decision", "fecha_reporte": "fecha_decision"})
    decisiones["saldo_cura_observado"] = decisiones["saldo_decision"] * decisiones["label_cura_3m"]
    return decisiones.reset_index(drop=True)
