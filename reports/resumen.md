# Informe fuera de tiempo · simulación

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

- Por diseño, los préstamos se originan el **1 de enero de su cosecha**
  (2019, 2020, 2021 o 2022): convención artificial de la simulación.
- Entrenamiento: cosechas 2019–2020; elección de peso: validación 2021;
  prueba intacta: cosecha **2022**.
- Las etiquetas de cada cosecha quedan observadas hasta diciembre de su propio
  año. No existe solapamiento cronológico entre entrenamiento, validación y test.
- Las variables se calculan usando filas **hasta el cierre del mes de decisión**.
- Pesos de la combinación: **0.75** logística + **0.25** boosting,
  elegidos por menor Brier en **2021**, sin ver las etiquetas de test.

## Métricas en test 2022 (simuladas)

| Modelo | AUC ROC | KS | PR-AUC/AP | Brier ↓ | Captura top 20% | Δ vs. mejor regla simple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistica | 0.637 | 0.229 | 0.804 | 0.196 | 40.87% | -0.22 pp |
| gradient_boosting | 0.615 | 0.202 | 0.796 | 0.200 | 41.43% | +0.34 pp |
| combinacion | 0.637 | 0.232 | 0.808 | 0.196 | 41.02% | -0.07 pp |

## Referencias sin Machine Learning

| Regla (fijada antes de evaluar) | Captura top 20% |
| --- | ---: |
| `saldo_descendente` | 41.09% |
| `dpd_30_antes_60_desempate_saldo` | 40.13% |

La mejor regla sencilla es **`saldo_descendente`** (41.09%).
La mejora observada del modelo seleccionado es **+0.34 puntos
porcentuales**. **No es uplift causal ni resultado económico**:
es la diferencia de captura observada en esta muestra sintética.
La diferencia es pequeña y no justifica por sí sola la complejidad de ML. Faltan intervalos de incertidumbre, costos y una evaluación prospectiva antes de recomendar su despliegue.

**Ganador entre los modelos en este experimento: `gradient_boosting`.** Se define a priori como el
modelo que concentra más saldo de cuentas que curan al priorizar el 20% de
cuentas por `P(cura) × saldo_decision`; el Brier es el desempate. No se
presupone que un ensemble mejore al modelo individual.

En el test hay **1483** cuentas elegibles;
prevalencia observada de cura: **71.5%**.
Los números son específicos de estas reglas artificiales, no transferibles.

## Dónde mirar

- [`metricas.json`](metricas.json): métricas, pesos y tamaños.
- [`tabla_lift.csv`](tabla_lift.csv): lift por decil para los tres modelos.
- [`figures/discriminacion.png`](figures/discriminacion.png): ROC/PR.
- [`figures/calibracion.png`](figures/calibracion.png): calibración observada.
- [`figures/lift.png`](figures/lift.png): lift por decil.
- [`figures/valor_capturado.png`](figures/valor_capturado.png): exposición
  de cuentas que curan, ordenando por valor esperado proxy.
- [`figures/valor_incremental.png`](figures/valor_incremental.png):
  comparación contra reglas sin ML y diferencias en puntos porcentuales.

## Límites antes de trabajar con clientes reales

Validar definiciones con Cobranza y Riesgo, revisar congelación histórica del
origen de datos, comparar contra política vigente, costos reales de contacto,
capacidad operativa, causas de mora, diferencias regulatorias, posibles sesgos,
escenarios de deriva (PSI), alertas de recalibración y ejecución/monitoreo.
Un score de curación **no demuestra que contactar una cuenta cause su curación**:
se requiere prueba de uplift o experimento para medir impacto incremental.
