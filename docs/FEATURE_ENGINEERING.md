# Variables que construimos con la historia de pagos

El modelo recibe una foto tomada después del cierre del mes de decisión, al que llamamos `t`. Para ese momento ya conocemos el pago, el saldo y el bucket de mora. En `construir_variables` unimos el panel con las decisiones y filtramos `mes_edad <= mes_decision` antes de calcular cualquier agregado. Las ventanas terminan en `t`; no usamos valores futuros ni ventanas centradas.

Hay una prueba que modifica deliberadamente las filas posteriores y comprueba que los predictores sigan siendo idénticos.

## Qué variables usamos y para qué

Empezamos por las condiciones del préstamo. `principal_original`, `plazo_meses`, `tasa_anual`, `grado_riesgo` y `proposito` describen el contrato y el segmento de partida. Para poner la cuota en contexto utilizamos `ingreso_mensual`, `dti_originacion` y `cuota_sobre_ingreso`. Así podemos observar la presión de las obligaciones sobre el ingreso.

El tamaño de la exposición al decidir se representa con `saldo_decision`, `ratio_saldo_original` y `mes_edad`. A eso sumamos `saldo_vencido_decision` y `cuotas_vencidas_equivalentes`, porque el monto vencido de la simulación aporta algo que el bucket de mora por sí solo no cuenta.

La situación actual está en `dias_mora_decision` y `pago_ratio_ultimo_mes`. Para mirar cómo venía pagando la cuenta calculamos `pago_ratio_media_3m`, `pago_ratio_volatilidad_3m` y `pago_ratio_tendencia_3m`. Buscamos distinguir pagos persistentes, irregulares y cambios recientes.

El historial de atrasos se resume con `meses_en_mora_3m`, `meses_en_mora_6m`, `peor_mora_6m` y `dpd_tendencia_3m`. Finalmente, `recencia_ultimo_al_dia_6m` y `ratio_pago_vs_cuota_6m` ayudan a describir cuánto hace que la cuenta estuvo al día y cómo se comportan los pagos frente a las cuotas.

En `recencia_ultimo_al_dia_6m` usamos el valor 7 cuando no aparece un mes al día en los últimos seis. Es un centinela numérico, no una afirmación de que el cliente lleve siete meses sin pagar. El ratio entre pago y cuota puede superar 1 si el simulador registra un abono mayor que la cuota.

## Dónde cortamos para evitar fugas

El modelo puede usar las condiciones del contrato desde la originación y el desempeño conocido hasta `t`. `saldo_decision` y `saldo_vencido_cierre` están disponibles al terminar ese mes; el segundo se renombra en la foto de decisión. Para aceptar una cuenta exigimos tres cierres anteriores consecutivos, completos y con datos no nulos. Si falta un mes, no lo sustituimos por otro.

Ninguna fila posterior a `t` entra en los predictores. Eso incluye moras, pagos, saldos, estados, fechas y derivados; solo se consultan después para construir `Y_i(t)`. También quedan fuera `label_cura_3m` y `saldo_cura_observado`, porque dependen del desenlace futuro.

`loan_id`, `fecha_decision` y `cosecha` se conservan como metadatos para identificar la observación y separar la validación, no como variables del estimador. `mes_edad` sí forma parte de la información histórica. El shock aleatorio y la resiliencia latente existen dentro del generador, pero no se exportan al panel.

Los modelos solo reciben las columnas de la lista positiva `COLUMNAS_MODELO` de `src/features/construccion.py`. Agregar una columna nueva al CSV no la convierte automáticamente en predictor.

## Qué produce el simulador

`src/data/generador.py` usa una semilla fija y cuatro cohortes de 2.600 créditos. Los ingresos se generan con una distribución lognormal acotada, el DTI con una beta acotada, los grados van de A a E, los principales son lognormales, los plazos son discretos y las tasas dependen del grado con algo de ruido.

Los pagos son aleatorios dentro de reglas definidas, pero la mora no se sortea. Cada mes se calcula `exigible = atraso_al_inicio + cuota`. Después de pagar queda `atraso_al_cierre = max(0, exigible − pago)`. El bucket es `min(120, 30 × ceil(atraso_al_cierre / cuota))`, con un pequeño margen para redondeos. El 120 es un tope, no un estado absorbente.

La cantidad pagada depende del contrato, la vulnerabilidad, un shock mensual y el atraso previo. Para 2022 se introduce una tensión macroeconómica inventada. El saldo principal y los intereses siguen reglas simplificadas. Las cuotas impagas equivalentes no reproducen la edad contractual exacta de una obligación vencida.

No usamos parámetros estimados con datos de una institución real. Este panel sirve para enseñar la trazabilidad, las pruebas contra fuga de información y la evaluación del ranking, no para concluir qué variables causan la recuperación de una cartera.
