# Modelo 01 · Regresión logística

**Función:** establecer una referencia probabilística sencilla e interpretable.
No se presupone que un método más complejo sea superior.

- Predictores: únicamente `COLUMNAS_MODELO`, nunca el target ni campos futuros.
- Transformación: mediana y estandarización para numéricas, one-hot para
  `grado_riesgo` y `proposito`.
- Algoritmo: `sklearn.linear_model.LogisticRegression`.
- Parámetros: [`hiperparametros.yaml`](hiperparametros.yaml).
- Entrenamiento/validación/test: [`docs/VALIDACION.md`](../../../docs/VALIDACION.md).

El coeficiente es interpretativo **en escala transformada**, no causal:
correlación entre una variable y curación no significa que cambiarla
altere las posibilidades reales de recuperación. La calibración se mide,
no se asume.
