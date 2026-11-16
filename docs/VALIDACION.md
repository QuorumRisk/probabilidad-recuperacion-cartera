# Cómo probamos las decisiones fuera del entrenamiento

Aquí sí hay separación temporal por cosecha. Los préstamos se originan el 1 de enero de 2019, 2020, 2021 o 2022, según su grupo. Las decisiones de cada cosecha ocurren de abril a septiembre y los resultados se observan como máximo hasta diciembre de ese mismo año.

La división se hace así. Los préstamos de 2019 y 2020 sirven para el entrenamiento inicial. Con 2021 elegimos el peso de la combinación usando Brier. Después reentrenamos los dos modelos con 2019, 2020 y 2021. La cosecha de 2022 queda reservada para la prueba final y no se usa para volver a ajustar parámetros.

De esa forma las etiquetas del entrenamiento terminan antes de que comience la etapa siguiente. No se usa `train_test_split` ni validación cruzada aleatoria. `run_demo.py` comprueba la separación y los tests revisan que las ventanas no se solapen.

## Los modelos que comparamos

La regresión logística es nuestra referencia probabilística sencilla. Imputa datos faltantes, escala las variables numéricas y codifica las categóricas con one-hot.

`HistGradientBoosting` utiliza imputación y one-hot también, pero puede aprender interacciones no lineales. Sus hiperparámetros están en YAML y no se retocan viendo el test.

La tercera opción combina las probabilidades con `w × P(logística) + (1−w) × P(boosting)`. Probamos `w = 0.25`, `0.50` y `0.75` en la validación de 2021 y elegimos por Brier. El peso queda fijo antes de evaluar 2022. Combinar no garantiza mejorar.

Un proceso productivo necesitaría varias ventanas de validación, intervalos de incertidumbre, reglas de negocio contrastadas y un gobierno de cambios.

## Primero, dos formas de trabajar sin ML

Antes de decidir que un score aporta algo, comparamos dos reglas que no miran el desenlace futuro:

- `saldo_descendente` ordena primero los créditos con mayor saldo al cierre de `t`.
- `dpd_30_antes_60_desempate_saldo` pone primero las cuentas de 30 DPD, luego las de 60, y dentro de cada grupo ordena por saldo.

Aplicamos a ambas el mismo criterio que a los modelos: seleccionar el 20% de las cuentas y medir qué porcentaje del saldo de las cuentas que después se pusieron al día quedó dentro de esa selección. Cada candidato se compara con la mejor de las dos reglas.

Las cifras están en `reports/metricas.json`, `reports/resumen.md` y `reports/figures/valor_incremental.png`. La diferencia es una mejora descriptiva en concentración. No mide el efecto causal de una llamada, los costos o la respuesta inducida por la cobranza.

## Qué medimos

AUC ROC muestra si el ranking sitúa las cuentas que curan por encima de las que no curan. KS mide la mayor distancia entre las tasas acumuladas TPR y FPR. PR-AUC, calculada como average precision, se fija en el ranking de la clase positiva y debe leerse junto con su prevalencia.

Con Brier, log loss y la curva de calibración vemos si las probabilidades estimadas guardan relación con las frecuencias observadas. También publicamos precisión y recall al umbral 0,50; es una lectura operativa ilustrativa, no un umbral óptimo de negocio.

El lift por decil muestra la concentración de cuentas que curan al ordenar por `P(cura)`. La curva de valor capturado mira la exposición usando `P(cura) × saldo(t)`. Y la captura top 20% permite comparar el score con ordenar por saldo o por el estado 30/60 DPD.

Para llamar ganador a un candidato usamos la mayor captura de saldo de las cuentas que curaron dentro del 20% priorizado por `P(cura) × saldo(t)` en el test 2022. Si hay empate, decide el Brier más bajo. Es un criterio declarado para el experimento; nombrar un ganador después de la prueba no autoriza a volver a ajustarlo con esa misma muestra.

## Lo que todavía faltaría con datos reales

La captura es un indicador sintético, no dinero recuperado. Para tomar decisiones operativas necesitaríamos incorporar el costo de gestión, la capacidad del equipo, el saldo vencido, la probabilidad de pago sin contacto y las restricciones regulatorias. Medir beneficio incremental requeriría además una prueba causal o experimental.

Tendríamos que auditar cuándo estuvo disponible cada campo, qué cambios se aplicaron de forma retroactiva y si existen fugas indirectas. También acordar cómo tratar prepago, venta, castigo o fallecimiento, y monitorear distribución, PSI, AUC, PR-AUC y calibración por ventana y segmento. Habría que definir responsables y límites, revisar sesgos, privacidad y seguridad, y comprobar que el proceso de cobranza se mantenga estable antes de atribuir cambios al modelo.
