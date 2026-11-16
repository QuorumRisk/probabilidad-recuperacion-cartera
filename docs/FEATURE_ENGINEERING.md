# Ingeniería de variables · historia hasta t

## Corte temporal y regla de oro

La foto se toma **después del cierre de `t`**: el pago, saldo y bucket de mora
de ese mes ya están asentados. La función `construir_variables` une el panel
con las decisiones y filtra **`mes_edad <= mes_decision` ANTES de calcular
agregados**. Las ventanas móviles terminan en `t`; no utilizan estadísticas
centradas ni valores posteriores. La prueba de invariancia modifica filas
futuras y exige exactamente los mismos predictores.

## Bloques y criterio de negocio

| Bloque | Variables concretas | Hipótesis ilustrativa |
| --- | --- | --- |
| Origen del contrato | `principal_original`, `plazo_meses`, `tasa_anual`, `grado_riesgo`, `proposito` | Condiciones de partida y segmento |
| Capacidad de pago | `ingreso_mensual`, `dti_originacion`, `cuota_sobre_ingreso` | Presión de obligaciones sobre liquidez |
| Exposición al decidir | `saldo_decision`, `ratio_saldo_original`, `mes_edad` | Magnitud y madurez del crédito |
| Atraso exigible al decidir | `saldo_vencido_decision`, `cuotas_vencidas_equivalentes` | Monto real vencido de la simulación, distinto de solo el bucket |
| Estado actual | `dias_mora_decision`, `pago_ratio_ultimo_mes` | Severidad y señal reciente de esfuerzo de pago |
| Dinámica de pagos (3m) | `pago_ratio_media_3m`, `pago_ratio_volatilidad_3m`, `pago_ratio_tendencia_3m` | Persistencia, irregularidad, cambio en pagos |
| Historial de mora | `meses_en_mora_3m`, `meses_en_mora_6m`, `peor_mora_6m`, `dpd_tendencia_3m` | Frecuencia y dirección del deterioro |
| Recencia | `recencia_ultimo_al_dia_6m`, `ratio_pago_vs_cuota_6m` | Cuánto hace que estuvo al día y cumplimiento agregado |

`recencia_ultimo_al_dia_6m` toma valor **7** si no aparece un mes al día
en los últimos seis: es una categoría numérica centinela, no una afirmación
de que llevaba siete meses sin pagar. El pago sobre cuota puede superar 1
si el simulador observa abonos superiores a la cuota.

## Política de anti-fuga

**Permitido:** columnas contractuales desde originación y desempeño
acumulado/observado **solo hasta `t`**. Los valores `saldo_decision` y
`saldo_vencido_cierre` son legítimos porque se conocen al cerrar el mes;
el segundo se renombra en la foto de decisión. Para entrar en el universo
se exigen tres cierres previos **consecutivos y con datos no nulos**; la
ausencia de un mes no se reemplaza por el mes vecino.

**Excluido como predictor:**

- **Todas las filas después de `t`**, cualquiera que sea su campo: mora,
  pagos, saldo, estado, fecha y derivados; únicamente se consultan al
  generar `Y_i(t)`.
- `label_cura_3m` y `saldo_cura_observado` (conocidos solo a posteriori).
- `loan_id`, `fecha_decision`, `cosecha` (metadatos para corte y validación;
  no son variables del estimador). El `mes_edad` sí es un predictor histórico.
- Shock aleatorio y resiliencia latente: existen dentro del generador,
  pero **no se exportan al panel**.
- Todo campo arbitrario que alguien agregue al CSV: los modelos consumen
  únicamente `COLUMNAS_MODELO` desde `src/features/construccion.py`.

## El generador NO es una estimación de una cartera real

El script `src/data/generador.py` usa semilla fija y cuatro cohortes
de 2.600 créditos cada una; ingresos lognormales acotados, DTI beta acotado,
grados A–E, principales lognormales, plazo discreto y tasa dependiente del
grado con ruido. **Los pagos son estocásticos; la mora no se asigna por
sorteo**. Cada mes: `exigible = atraso_al_inicio + cuota`; tras aplicar
el pago, `atraso_al_cierre = max(0, exigible − pago)`. El bucket es
`min(120, 30 × ceil(atraso_al_cierre / cuota))`, con margen numérico para
redondeos. El de 120 es un tope, **no absorbente**. La función de pago
depende de contrato, vulnerabilidad, shock mensual y atraso previo; se
añade en 2022 una tensión macro inventada. Los equivalentes de cuotas
impagas **no reproducen la antigüedad contractual exacta de un impago**.
El saldo principal e intereses se aproximan con reglas simplificadas.

Este panel no tiene estimación paramétrica proveniente de entidades reales
ni permite concluir qué señales causan curación. El experimento sirve para
mostrar trazabilidad, prueba de fuga y evaluación de predicciones.
