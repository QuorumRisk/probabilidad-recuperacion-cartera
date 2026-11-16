"""Panel de préstamos enteramente sintético con libro de cuotas vencidas.

La incertidumbre modifica cuánto dinero paga cada préstamo, **no** su DPD.
El atraso monetario es una obligación exigible: pendiente anterior + cuota
del mes - pago. El bucket se deriva del saldo vencido expresado en cuotas.
Ningún registro representa a una persona o una cartera real.
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
    """Genera 12 cierres mensuales por crédito originado en enero de su cosecha.

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

        saldo_vencido = np.zeros(n, dtype=float)
        saldo = principal.copy()
        interes_pendiente = np.zeros(n, dtype=float)
        for mes in range(1, meses + 1):
            # Condiciones de entorno ilustrativas; 2022 presenta leve tensión.
            tension = {2019: -0.17, 2020: 0.07, 2021: -0.08, 2022: 0.22}.get(cosecha, 0.0)
            shock = rng.normal(0, 0.85, n)
            vencido_inicio = saldo_vencido.copy()
            cuota_redondeada = np.round(cuota, 2)
            exigible = np.round(vencido_inicio + cuota_redondeada, 2)

            # Cliente sano: pagar la cuota íntegra es el estado habitual;
            # el impago parcial nace de capacidad, exposición y shocks.
            al_dia = vencido_inicio <= 0.0
            p_cuota = _sigmoide(
                2.85 - 0.72 * vulnerabilidad + 0.23 * resiliencia_oculta
                - 0.52 * shock - 0.40 * tension
            )
            # Cliente moroso: ponerse al día exige recursos para cuota actual
            # y todas las cuotas vencidas. Un atraso grande reduce probabilidad.
            cuotas_atrasadas = vencido_inicio / cuota_redondeada
            p_regularizar = _sigmoide(
                0.35 - 0.52 * vulnerabilidad + 0.24 * resiliencia_oculta
                - 0.54 * shock - 0.28 * tension - 0.34 * cuotas_atrasadas
            )
            decide_pagar = rng.random(n)
            pago_parcial = np.round(cuota_redondeada * rng.beta(1.5, 2.5, n), 2)
            pago_con_atraso = np.round(cuota_redondeada * rng.uniform(0.05, 1.25, n), 2)
            pago = np.where(
                al_dia,
                np.where(decide_pagar < p_cuota, cuota_redondeada, pago_parcial),
                np.where(decide_pagar < p_regularizar, exigible, pago_con_atraso),
            )
            # Sin prepagos ni condonaciones en este caso: no puede cobrarse
            # más del monto exigible y toda obligación impaga queda en mora.
            pago = np.minimum(np.maximum(0.0, pago), exigible)
            saldo_vencido = np.round(np.maximum(0.0, exigible - pago), 2)
            cuotas_impagas_equivalentes = np.ceil(
                np.maximum(0.0, saldo_vencido / cuota_redondeada - 1e-10)
            ).astype(int)
            dpd = np.minimum(cuotas_impagas_equivalentes, 4) * 30

            # Principal simplificado: intereses exigibles se pagan antes que
            # principal. El libro contractual de cuotas vencidas anterior es
            # la fuente única del DPD, no el saldo principal contable.
            interes_pendiente = np.round(interes_pendiente + saldo * tasa_mes, 2)
            pago_interes = np.minimum(pago, interes_pendiente)
            interes_pendiente = np.round(interes_pendiente - pago_interes, 2)
            saldo = np.maximum(0.0, np.round(saldo - (pago - pago_interes), 2))
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
                        "cuota_programada": cuota_redondeada,
                        "saldo_vencido_inicio": vencido_inicio,
                        "obligacion_exigible": exigible,
                        "pago_realizado": pago,
                        "saldo_vencido_cierre": saldo_vencido.copy(),
                        "saldo_actual": saldo.copy(),
                        "dias_mora": dpd.copy(),
                    }
                )
            )

    panel = pd.concat(bloques, ignore_index=True)
    return panel.sort_values(["loan_id", "mes_edad"], kind="stable").reset_index(drop=True)
