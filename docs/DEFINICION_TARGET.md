# Cómo definimos el resultado que queremos anticipar

Este documento describe el target del panel sintético. En un trabajo con datos reales lo acordaríamos primero con Riesgo, Cobranza, Contabilidad y Operaciones. La definición no se decide únicamente desde el equipo que entrena el modelo.

## La cuenta y el momento de decisión

Tomamos una sola observación por préstamo: la primera vez que cumple las condiciones para entrar al ejercicio. El momento `t` es el último día de un mes cuyo historial ya está cerrado y disponible. Usar varias fotos de la misma cuenta como si fueran independientes podría inflar las métricas.

Los préstamos simulados nacen el 1 de enero de su respectivo año, 2019, 2020, 2021 o 2022. Las decisiones se toman entre los meses de edad 4 y 9, es decir, entre abril y septiembre de cada cosecha.

## Qué cuentas entran

En el cierre `t` la cuenta debe tener saldo positivo y un `dias_mora` de 30 o 60. Son buckets calculados con cuotas impagas equivalentes, no días exactos transcurridos desde la primera obligación vencida.

También pedimos tres cierres anteriores consecutivos (`t−3`, `t−2` y `t−1`). Deben existir los datos de pago, cuota, saldo, fecha y mora, además de los campos observados al decidir. Para conocer el resultado hacen falta los tres cierres posteriores (`t+1`, `t+2` y `t+3`), igualmente consecutivos y con mora no nula.

No incluimos cuentas sanas al corte ni cuentas en 90 o 120 DPD. Tampoco entran registros sin saldo, con meses faltantes o con la misma clave crédito-mes duplicada. En el simulador, 120 es el bucket superior pero no un estado absorbente: una cuenta puede salir de ahí si paga lo necesario. En una cartera real habría que definir por separado castigos y otros estados terminales.

## Qué significa que una cuenta vuelva al día

El evento positivo es `Y_i(t) = 1` cuando existe algún `k ∈ {1,2,3}` para el cual `dias_mora_i(t+k) = 0`. Si ninguno de los tres cierres posteriores llega a cero, la etiqueta es `Y_i(t) = 0`.

La ventana empieza después de la decisión y dura tres cierres mensuales. Bajar de 60 a 30 DPD es una mejora, pero no cuenta como volver al día. Y si una cuenta llega a cero en `t+1` y vuelve a atrasarse en `t+3`, queda positiva: aquí medimos si se puso al día alguna vez dentro del horizonte, no si permaneció así. Es una distinción que tendríamos que discutir con el negocio.

Un ejemplo ayuda. Supongamos que una cuenta tuvo mora 0 en `t−2`, 30 en `t−1` y 60 al decidir en `t`. Luego registra 60 en `t+1`, 0 en `t+2` y 30 en `t+3`. Su etiqueta es 1 porque estuvo al día en `t+2`, aunque haya reincidido. Los datos de `t−2` y `t−1` son historia observada; el límite de las variables del modelo es `t`. Los meses posteriores solo sirven para construir la etiqueta.

En el código, `dias_mora(t+1..t+3)` se consulta únicamente al etiquetar. Ningún campo posterior a `t` entra en los predictores.

## Qué hacemos si falta información

Una cuenta sin los tres cierres futuros, o con `dias_mora` en `NaN` en cualquiera de ellos, no puede etiquetarse. Se excluye, nunca se trata automáticamente como un caso negativo. Lo mismo ocurre si no tiene los tres cierres anteriores completos y consecutivos.

El panel de prueba mantiene doce cierres por préstamo. No modela venta de cartera, prepago, reestructuración, modificación, castigo, fallecimiento o errores operativos. Con datos reales esas situaciones necesitarían reglas propias y posiblemente un análisis de riesgos competitivos.

## Cómo interpretamos el saldo capturado

Usamos `saldo_curado_i = saldo_i(t) × Y_i(t)`. Es la exposición que tenía la cuenta al decidir, siempre que después haya llegado a cero DPD. No son pesos efectivamente cobrados ni valor contable realizado.

Para priorizar se utiliza `p_i × saldo_i(t)` como aproximación sencilla al valor esperado. En un encargo real entrarían el monto vencido, la severidad, los costos de gestión, la probabilidad de respuesta incremental, los flujos de recuperación y las restricciones de contacto.
