# Regresión logística

Empezamos con una referencia probabilística sencilla e interpretable. No hay motivo para dar por hecho que un método más complejo vaya a superarla.

El modelo recibe únicamente las columnas de `COLUMNAS_MODELO`, sin el target ni información de meses futuros. Para los predictores numéricos se usa imputación por mediana y estandarización. `grado_riesgo` y `proposito` se codifican con one-hot. El estimador es `sklearn.linear_model.LogisticRegression`, con la configuración de [`hiperparametros.yaml`](hiperparametros.yaml).

La separación de entrenamiento, validación y test está documentada en [`docs/VALIDACION.md`](../../../docs/VALIDACION.md). Los coeficientes pueden ayudar a interpretar asociaciones en la escala transformada, pero no prueban causalidad. Cambiar una variable asociada con curación no implica que cambie la posibilidad real de recuperación. También medimos calibración, porque no basta con asumirla.
