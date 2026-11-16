"""Panel de préstamos *enteramente sintético*; ningún registro representa a personas.

Todas las transiciones se simulan en orden cronológico con un RNG de semilla fija.
El estado del mes depende de historial/ingresos/carga financiera y de choques
no observables por el modelo. No se introduce una etiqueta prefabricada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEMILLA = 20261116
COSECHAS = (2019, 2020, 2021, 2022)


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generar_panel(
    semilla: int = SEMILLA,
    prestamos_por_cosecha: int = 2600,
    cosechas: tuple[int, ...] = COSECHAS,
    meses: int = 12,
) -> pd.DataFrame:
    """Genera 12 observaciones mensuales por crédito originado en enero.

    Los meses 4–9 contienen las decisiones elegibles; su horizonte de tres
    meses termina, como máximo, en diciembre del mismo año. Esto permite
    separar íntegramente las decisiones y sus desenlaces por cosecha anual.
    """
    if meses != 12:
        raise ValueError("El experimento cronológico exige 12 meses por cosecha.")
    if prestamos_por_cosecha < 1 or not cosechas:
        raise ValueError("Es necesario generar préstamos y al menos una cosecha.")

    rng = np.random.default_rng(semilla)
    bloques: list[pd.DataFrame] = []
    propositos = np.array(["consumo", "negocio", "educacion", "vehiculo", "hogar"])

    for cosecha in cosechas:
        n = prestamos_por_cosecha
        ids = np.array([f"SIM-{cosecha}-{i:06d}" for i in range(n)])
        ingresos = np.clip(rng.lognormal(np.log(36000), 0.48, n), 15000, 190000)
        dti = np.clip(rng.beta(2.8, 4.5, n) * 0.82, 0.04, 0.78)
        grado_num = np.clip(
            np.rint(rng.normal(1.50 + 2.0 * (dti - 0.30), 1.0, n)).astype(int),
            0,
            4,
        )
        grado = np.array(list("ABCDE"))[grado_num]
        proposito = rng.choice(propositos, n, p=[0.40, 0.20, 0.08, 0.18, 0.14])
        principal = np.clip(rng.lognormal(np.log(14500), 0.62, n), 2000, 85000)
        plazo = rng.choice([24, 36, 48], n, p=[0.25, 0.56, 0.19])
        tasa = np.clip(0.10 + grado_num * 0.036 + rng.normal(0, 0.022, n), 0.07, 0.38)
        tasa_mes = tasa / 12.0
        cuota = principal * tasa_mes / (1.0 - (1.0 + tasa_mes) ** (-plazo))
        ratio_cuota_ingreso = cuota / (ingresos / 12.0)
        resiliencia_oculta = rng.normal(0, 0.8, n)
        vulnerabilidad = (
            0.52 * grado_num
            + 1.80 * (dti - 0.28)
            + 1.1 * (ratio_cuota_ingreso - 0.24)
            - 0.32 * resiliencia_oculta
        )

        dpd = np.zeros(n, dtype=int)
        saldo = principal.copy()
        for mes in range(1, meses + 1):
            # Condiciones de entorno ilustrativas; 2022 presenta leve tensión.
            tension = {2019: -0.17, 2020: 0.07, 2021: -0.08, 2022: 0.22}.get(cosecha, 0.0)
            shock = rng.normal(0, 0.85, n)
            pago_ratio = np.clip(
                rng.normal(
                    1.10
                    - 0.34 * (dpd > 0)
                    - 0.075 * vulnerabilidad
                    + 0.16 * resiliencia_oculta
                    - 0.22 * shock
                    - 0.08 * tension,
                    0.25,
                    n,
                ),
                0.0,
                2.2,
            )
            u = rng.random(n)
            siguiente = dpd.copy()

            al_dia = dpd == 0
            p_entrada = _sigmoide(
                -3.05 + 0.64 * vulnerabilidad + 0.50 * shock + 0.40 * tension
                - 1.30 * (pago_ratio - 0.85)
            )
            siguiente[al_dia & (u < p_entrada)] = 30

            mora30 = dpd == 30
            p_cura30 = _sigmoide(
                0.12 - 0.48 * vulnerabilidad + 1.8 * (pago_ratio - 0.72)
                + 0.18 * resiliencia_oculta - 0.32 * shock - 0.3 * tension
            )
            p_empeora30 = _sigmoide(
                -0.70 + 0.32 * vulnerabilidad - 1.2 * (pago_ratio - 0.65)
            )
            siguiente[mora30 & (u < p_cura30)] = 0
            siguiente[
                mora30 & (u >= p_cura30) & (u < p_cura30 + (1 - p_cura30) * p_empeora30)
            ] = 60

            mora60 = dpd == 60
            p_cura60 = _sigmoide(
                -0.85 - 0.44 * vulnerabilidad + 1.55 * (pago_ratio - 0.70)
                + 0.17 * resiliencia_oculta - 0.36 * shock
            )
            p_mejora60 = 0.34
            p_empeora60 = 0.37
            siguiente[mora60 & (u < p_cura60)] = 0
            siguiente[mora60 & (u >= p_cura60) & (u < p_cura60 + (1 - p_cura60) * p_mejora60)] = 30
            siguiente[
                mora60
                & (u >= p_cura60 + (1 - p_cura60) * p_mejora60)
                & (u < p_cura60 + (1 - p_cura60) * (p_mejora60 + p_empeora60))
            ] = 90

            mora90 = dpd == 90
            p_cura90 = _sigmoide(-2.9 - 0.5 * vulnerabilidad + 0.7 * (pago_ratio - 0.7))
            siguiente[mora90 & (u < p_cura90)] = 0
            siguiente[mora90 & (u >= p_cura90) & (u < p_cura90 + 0.18)] = 60
            siguiente[mora90 & (u >= p_cura90 + 0.18) & (u < p_cura90 + 0.52)] = 120
            # 120 DPD es absorbente en esta simulación simplificada.
            siguiente[dpd == 120] = 120
            dpd = siguiente

            pago = cuota * pago_ratio
            interes = saldo * tasa_mes
            saldo = np.maximum(0.0, saldo - np.maximum(0.0, pago - interes))
            bloques.append(
                pd.DataFrame(
                    {
                        "loan_id": ids,
                        "cosecha": cosecha,
                        "fecha_originacion": pd.Timestamp(cosecha, 1, 1),
                        "fecha_reporte": pd.Timestamp(cosecha, mes, 1) + pd.offsets.MonthEnd(0),
                        "mes_edad": mes,
                        "ingreso_anual": ingresos,
                        "dti_originacion": dti,
                        "grado_riesgo": grado,
                        "proposito": proposito,
                        "principal_original": principal,
                        "plazo_meses": plazo,
                        "tasa_anual": tasa,
                        "cuota_programada": cuota,
                        "pago_realizado": pago,
                        "saldo_actual": saldo.copy(),
                        "dias_mora": dpd.copy(),
                    }
                )
            )

    panel = pd.concat(bloques, ignore_index=True)
    return panel.sort_values(["loan_id", "mes_edad"], kind="stable").reset_index(drop=True)
