# Validación temporal y criterios de evaluación

## Separación innegociable

| Etapa | Cosechas de originación | Uso permitido |
| --- | --- | --- |
| Entrenamiento inicial | 2019 y 2020 | Ajustar logística y boosting |
| Validación | 2021 | Escoger peso de combinación por Brier |
| Reentrenamiento | 2019–2021 | Ajustar los dos candidatos finales |
| **Test final** | **2022** | Reportar resultados; no reoptimizar parámetros |

Las fechas no solo están separadas por cosecha. **Todas las originaciones
son en enero y las decisiones de cada cosecha ocurren de abril a septiembre**.
Las etiquetas terminan entre julio y diciembre del mismo año, de manera
que ningún desenlace de entrenamiento llega al año siguiente. Esto evita
solapar cronológicamente etiquetas con la siguiente fase.

No se utiliza `train_test_split` ni CV aleatorio. La separación se verifica
en `run_demo.py` y las pruebas unitarias comprueban la regla temporal.

## Tres candidatas sin suponer que ensemble gana

1. **Logística** con imputación, escala de numéricas y codificación one-hot
   de categorías: baseline transparente.
2. **HistGradientBoosting** con imputación y one-hot: interacciones no
   lineales. Hiperparámetros definidos en YAML, sin ajuste al test.
3. **Combinación** `w × P(logística) + (1−w) × P(boosting)`; se prueban
   `w = 0.25, 0.50, 0.75` con **Brier en validación 2021**. Después el peso
   queda congelado y se evalúa 2022. No equivale a una garantía de mejora.

En un proceso productivo haría falta validar múltiples ventanas, intervalos
de incertidumbre, benchmark de reglas de negocio y gobierno de cambios.

## Qué se mide y para qué

| Métrica | Qué responde |
| --- | --- |
| AUC ROC | ¿Ordena curaciones por encima de no-curaciones? |
| KS | Máxima distancia entre las tasas acumuladas TPR y FPR |
| PR-AUC (average precision) | Calidad de ranking enfocada en la clase positiva; considerar prevalencia |
| Brier, log loss y curva de calibración | ¿Las probabilidades estimadas son coherentes con las frecuencias? |
| Precisión y recall con umbral 0,50 | Lectura operativa **ilustrativa**, no umbral óptimo de negocio |
| Lift por decil | Concentración de cuentas curadas ordenadas por `P(cura)` |
| Curva de saldo de cuentas que curan | Concentración de exposición al ordenar por `P(cura) × saldo(t)` |

**Criterio explícito para nombrar un ganador en el informe:** mayor captura
de saldo de las cuentas que curaron en el **top 20% de cuentas**, ordenadas
por `P(cura) × saldo(t)` en test 2022; empate por Brier más bajo. Nombrar
un ganador retrospectivo no autoriza a retocar el modelo contra test.

La métrica de captura es un **proxy sintético**: no representa ingresos de
cobranza ni estima causalidad. Una decisión real incluiría costo de gestión,
capacidad de equipo, saldo vencido, probabilidad de pago sin contacto,
restricciones normativas y test de incremento causal/uplift.

## Revisión adicional obligatoria en datos reales

- Auditoría de disponibilidad de cada campo al corte (latencia de sistemas,
  cambios retroactivos, reestructuraciones y posibles leakage indirectos).
- Censura y competing risks: venta, prepago, castigo, fallecimiento, etc.
- Monitoreo de distribución (por ejemplo, PSI), AUC/PR y calibración por
  ventana y subsegmento, con límites y responsables.
- Estabilidad del proceso de cobranza, sesgos, privacidad y seguridad de datos.
- Análisis de beneficio incremental antes de afirmar impacto económico.
