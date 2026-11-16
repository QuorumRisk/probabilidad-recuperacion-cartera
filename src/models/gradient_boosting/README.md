# Gradient boosting

Este candidato permite capturar relaciones no lineales e interacciones entre los pagos, la severidad de mora y la capacidad del cliente.

Usa la misma lista positiva de predictores que la logística. El preprocesamiento incluye imputación numérica y one-hot para las variables categóricas. El algoritmo es `sklearn.ensemble.HistGradientBoostingClassifier`; sus límites de complejidad están en [`hiperparametros.yaml`](hiperparametros.yaml).

Después del ajuste con las cosechas de 2019 a 2021 se evalúa exclusivamente en el test temporal de 2022. Un boosting puede ordenar mejor y calibrar peor, o puede ocurrir lo contrario. Lo decidimos con los resultados fuera de muestra y el criterio de priorización declarado antes de mirar el test, no por el nombre del algoritmo.
