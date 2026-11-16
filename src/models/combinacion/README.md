# Combinación de probabilidades

Aquí juntamos la probabilidad de la logística con la del gradient boosting. La fórmula es `p_comb = w × p_logistica + (1 − w) × p_gradient_boosting`.

Comparamos los pesos `0.25`, `0.50` y `0.75` usando Brier en la cosecha de validación de 2021. Una vez elegido el peso, se mantiene fijo al evaluar 2022. El resultado de test nunca interviene en esa elección.

No entrenamos un modelo adicional de stacking ni usamos predicciones de test para ajustar la mezcla. La combinación puede rendir peor que cualquiera de sus componentes. La comparación completa queda en `reports/metricas.json`.
