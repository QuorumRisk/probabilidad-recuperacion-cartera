"""Orquesta simulación, target, variables, validación temporal y reportes.

Uso: python run_demo.py
Pruebas: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generador import COSECHAS, SEMILLA, generar_panel
from src.data.preparacion import definir_observaciones
from src.evaluacion.graficos import guardar_figuras
from src.evaluacion.metricas import calcular_metricas, captura_top_20, tabla_lift
from src.features.construccion import COLUMNAS_MODELO, construir_variables
from src.models.combinacion.modelo import combinar, seleccionar_peso
from src.models.gradient_boosting.modelo import crear_modelo as crear_boosting
from src.models.logistica.modelo import crear_modelo as crear_logistica

RAIZ = Path(__file__).resolve().parent
ENTRENAMIENTO = (2019, 2020)
VALIDACION = 2021
TEST = 2022


def _generar_resumen(
    metricas: dict, tabla: pd.DataFrame, *, peso: float, ganador: str
) -> str:
    filas = []
    for nombre, m in metricas["modelos"].items():
        filas.append(
            f"| {nombre} | {m['auc_roc']:.3f} | {m['ks']:.3f} | "
            f"{m['pr_auc_average_precision']:.3f} | {m['brier']:.3f} | "
            f"{m['captura_saldo_curado_top_20_pct']:.1f}% |"
        )
    return f"""# Informe fuera de tiempo · simulación

> **⚠️ Datos exclusivamente sintéticos.** Estos números no prueban desempeño
> en clientes ni anticipan resultados comerciales. No contienen datos reales.

## Target y población

- Unidad: primera observación elegible por crédito, en meses de edad 4–9.
- Universo: saldo positivo y **30 o 60 días de mora** al cierre mensual.
- Target: **volver a 0 días de mora en cualquiera de los siguientes tres meses**.
- Exclusiones: otras moras al decidir, decisiones sin tres meses futuros completos,
  duplicados e insuficiente historial mínimo de tres meses.
- Saldo en decisión: proxy de exposición de cuentas que eventualmente curan,
  **no equivale a efectivo cobrado ni a recuperación contractual de principal**.

## Corte y validación

- Generación de préstamos originados en enero de cada año, 2019–2022.
- Entrenamiento: cosechas 2019–2020; elección de peso: validación 2021;
  prueba intacta: cosecha **2022**.
- Las etiquetas de cada cosecha quedan observadas hasta diciembre de su propio
  año. No existe solapamiento cronológico entre entrenamiento, validación y test.
- Las variables se calculan usando filas **hasta el cierre del mes de decisión**.
- Pesos de la combinación: **{peso:.2f}** logística + **{1-peso:.2f}** boosting,
  elegidos por menor Brier en **2021**, sin ver las etiquetas de test.

## Métricas en test 2022 (simuladas)

| Modelo | AUC ROC | KS | PR-AUC/AP | Brier ↓ | Saldo curado capturado, top 20% |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(filas)}

**Ganador en este experimento: `{ganador}`.** Se define a priori como el
modelo que concentra más saldo de cuentas que curan al priorizar el 20% de
cuentas por `P(cura) × saldo_decision`; el Brier es el desempate. No se
presupone que un ensemble mejore al modelo individual.

En el test hay **{metricas['muestras']['test_2022']}** cuentas elegibles;
prevalencia observada de cura: **{metricas['prevalencia_test']:.1%}**.
Los números son específicos de estas reglas artificiales, no transferibles.

## Dónde mirar

- [`metricas.json`](metricas.json): métricas, pesos y tamaños.
- [`tabla_lift.csv`](tabla_lift.csv): lift por decil para los tres modelos.
- [`figures/discriminacion.png`](figures/discriminacion.png): ROC/PR.
- [`figures/calibracion.png`](figures/calibracion.png): calibración observada.
- [`figures/lift.png`](figures/lift.png): lift por decil.
- [`figures/valor_capturado.png`](figures/valor_capturado.png): exposición
  de cuentas que curan, ordenando por valor esperado proxy.

## Límites antes de trabajar con clientes reales

Validar definiciones con Cobranza y Riesgo, revisar congelación histórica del
origen de datos, comparar contra política vigente, costos reales de contacto,
capacidad operativa, causas de mora, diferencias regulatorias, posibles sesgos,
escenarios de deriva (PSI), alertas de recalibración y ejecución/monitoreo.
Un score de curación **no demuestra que contactar una cuenta cause su curación**:
se requiere prueba de uplift o experimento para medir impacto incremental.
"""


def ejecutar(semilla: int = SEMILLA, prestamos_por_cosecha: int = 2600) -> dict:
    datos = RAIZ / "data"
    informes = RAIZ / "reports"
    datos.mkdir(parents=True, exist_ok=True)
    informes.mkdir(parents=True, exist_ok=True)

    panel = generar_panel(semilla=semilla, prestamos_por_cosecha=prestamos_por_cosecha)
    decisiones = definir_observaciones(panel)
    features = construir_variables(panel, decisiones)

    # Solo los resultados agregados, no el panel ni los scores individuales,
    # viajan al repositorio público. Los CSV intermedios viven en data/ ignorado.
    panel.to_csv(datos / "panel_sintetico.csv", index=False)
    decisiones.to_csv(datos / "decisiones_sinteticas.csv", index=False)

    tr = features.loc[features["cosecha"].isin(ENTRENAMIENTO)]
    va = features.loc[features["cosecha"].eq(VALIDACION)]
    te = features.loc[features["cosecha"].eq(TEST)]
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError("Falta una cosecha con observaciones elegibles.")
    if not (tr["fecha_decision"].max() < va["fecha_decision"].min() < te["fecha_decision"].min()):
        raise ValueError("Las ventanas de decisión se solapan temporalmente.")
    if not (tr["fecha_decision"].max() + pd.DateOffset(months=3) < va["fecha_decision"].min()):
        raise ValueError("Las etiquetas de entrenamiento pueden invadir validación.")
    if not (va["fecha_decision"].max() + pd.DateOffset(months=3) < te["fecha_decision"].min()):
        raise ValueError("Las etiquetas de validación pueden invadir test.")

    xtr, ytr = tr[COLUMNAS_MODELO], tr["label_cura_3m"].to_numpy()
    xva, yva = va[COLUMNAS_MODELO], va["label_cura_3m"].to_numpy()
    xte, yte = te[COLUMNAS_MODELO], te["label_cura_3m"].to_numpy()

    # Decisión del peso exclusivamente con cosecha 2021.
    provisional_logit = crear_logistica().fit(xtr, ytr)
    provisional_boost = crear_boosting().fit(xtr, ytr)
    pl_val = provisional_logit.predict_proba(xva)[:, 1]
    pb_val = provisional_boost.predict_proba(xva)[:, 1]
    peso_logistica, briers_validacion = seleccionar_peso(yva, pl_val, pb_val)

    # Reentrenamiento de ambos estimadores con 2019–2021; 2022 permanece intacto.
    ajuste = pd.concat([tr, va], ignore_index=True)
    logit = crear_logistica().fit(ajuste[COLUMNAS_MODELO], ajuste["label_cura_3m"])
    boosting = crear_boosting().fit(ajuste[COLUMNAS_MODELO], ajuste["label_cura_3m"])
    pl = logit.predict_proba(xte)[:, 1]
    pb = boosting.predict_proba(xte)[:, 1]
    predicciones = {
        "logistica": pl,
        "gradient_boosting": pb,
        "combinacion": combinar(pl, pb, peso_logistica),
    }
    saldos = te["saldo_decision"].to_numpy(dtype=float)
    resultados = {}
    lifts = []
    for nombre, prob in predicciones.items():
        met = calcular_metricas(yte, prob)
        met["captura_saldo_curado_top_20_pct"] = captura_top_20(yte, prob, saldos)
        resultados[nombre] = met
        lifts.append(tabla_lift(yte, prob, saldos, nombre))
    # Métrica de negocio explícita: capturar saldo curado en top 20%; empate por Brier.
    ganador = max(resultados, key=lambda n: (resultados[n]["captura_saldo_curado_top_20_pct"], -resultados[n]["brier"]))

    reporte = {
        "aviso": "SIMULACION. Datos sintéticos; ninguna métrica corresponde a clientes reales.",
        "semilla": semilla,
        "prestamos_por_cosecha": prestamos_por_cosecha,
        "cosechas": list(COSECHAS),
        "target": "0 dias de mora en alguno de los 3 meses posteriores a estar en 30/60 DPD",
        "corte": "ultimo dia del mes de decision; historia hasta t, target en t+1..t+3",
        "muestras": {"entrenamiento_2019_2020": len(tr), "validacion_2021": len(va), "test_2022": len(te)},
        "prevalencia_test": round(float(yte.mean()), 6),
        "peso_logistica_combinacion": peso_logistica,
        "brier_validacion_por_peso": briers_validacion,
        "criterio_ganador": "captura del saldo de cuentas curadas en top 20% por p(cura)*saldo al decidir; empate por Brier",
        "modelo_ganador": ganador,
        "modelos": resultados,
    }
    (informes / "metricas.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tabla = pd.concat(lifts, ignore_index=True)
    tabla.to_csv(informes / "tabla_lift.csv", index=False, float_format="%.4f")
    (informes / "resumen.md").write_text(
        _generar_resumen(reporte, tabla, peso=peso_logistica, ganador=ganador), encoding="utf-8"
    )
    guardar_figuras(yte, predicciones, saldos, informes / "figures")
    return reporte


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caso reproducible de curación de mora, 100% sintético.")
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--prestamos-por-cosecha", type=int, default=2600)
    args = parser.parse_args()
    reporte = ejecutar(semilla=args.semilla, prestamos_por_cosecha=args.prestamos_por_cosecha)
    print(json.dumps({"muestras": reporte["muestras"], "ganador": reporte["modelo_ganador"], "modelos": reporte["modelos"]}, indent=2, ensure_ascii=False))
