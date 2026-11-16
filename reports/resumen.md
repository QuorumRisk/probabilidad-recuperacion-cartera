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

- Generación de préstamos originados en enero de cada año, 2019–2022.
- Entrenamiento: cosechas 2019–2020; elección de peso: validación 2021;
  prueba intacta: cosecha **2022**.
- Las etiquetas de cada cosecha quedan observadas hasta diciembre de su propio
  año. No existe solapamiento cronológico entre entrenamiento, validación y test.
- Las variables se calculan usando filas **hasta el cierre del mes de decisión**.
- Pesos de la combinación: **0.75** logística + **0.25** boosting,
  elegidos por menor Brier en **2021**, sin ver las etiquetas de test.

## Métricas en test 2022 (simuladas)

| Modelo | AUC ROC | KS | PR-AUC/AP | Brier ↓ | Saldo curado capturado, top 20% |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistica | 0.673 | 0.258 | 0.745 | 0.219 | 42.1% |
| gradient_boosting | 0.652 | 0.247 | 0.734 | 0.227 | 42.7% |
| combinacion | 0.672 | 0.250 | 0.748 | 0.219 | 42.0% |

**Ganador en este experimento: `gradient_boosting`.** Se define a priori como el
modelo que concentra más saldo de cuentas que curan al priorizar el 20% de
cuentas por `P(cura) × saldo_decision`; el Brier es el desempate. No se
presupone que un ensemble mejore al modelo individual.

En el test hay **1217** cuentas elegibles;
prevalencia observada de cura: **60.6%**.
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
