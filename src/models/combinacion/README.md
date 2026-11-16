# Modelo 03 · Combinación evaluada

Combinación lineal de dos probabilidades:

`p_comb = w × p_logistica + (1 − w) × p_gradient_boosting`.

Los tres pesos candidatos (`0.25`, `0.50`, `0.75`) se comparan por
**Brier en la cosecha de validación 2021**. El peso elegido se congela
antes de estimar 2022. Nunca se usa el desenlace de test para escogerlo.

No hay apilado (stacking) ni otro modelo entrenado con predicciones de
test. **La combinación puede perder** frente a cualquiera de sus partes.
Su comparación completa figura en `reports/metricas.json`.
