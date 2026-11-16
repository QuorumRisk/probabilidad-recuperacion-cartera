# Modelo 02 · Gradient boosting

**Función:** capturar relaciones no lineales e interacciones entre
comportamiento, severidad de mora y capacidad de pago.

- Predictores: la misma lista blanca del modelo logístico.
- Transformación: imputación numérica y one-hot de variables categóricas.
- Algoritmo: `sklearn.ensemble.HistGradientBoostingClassifier`.
- Hiperparámetros limitadores de complejidad:
  [`hiperparametros.yaml`](hiperparametros.yaml).
- Evaluación exclusivamente en test OOT 2022 tras ajuste con 2019–2021.

El boosting puede discriminar mejor y estar peor calibrado, o lo contrario.
La selección no se basa en apariencia del algoritmo sino en resultados
fuera de muestra y criterio de priorización previamente declarado.
