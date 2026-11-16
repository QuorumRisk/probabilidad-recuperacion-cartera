# Definición del target · curación de mora en tres meses

> Definición **ilustrativa** sobre un panel sintético. En producción, la
> definición se valida con responsables de Riesgo, Cobranza, Contabilidad y
> Operaciones antes de escribir una sola línea de modelado.

## 1. Unidad y fecha de decisión

- Una observación es la **primera foto elegible de cada préstamo**.
- `t` = último día de un mes con historial cerrado y disponible.
- Para el experimento sintético, los préstamos de **cada cosecha** se
  originan el 1 de enero de **su respectivo año** (2019, 2020, 2021 o 2022)
  y se evalúan como máximo en meses de edad 4–9 (**abril–septiembre**).
- No se toma más de una decisión por crédito: se evita que múltiples fotos
  correlacionadas inflen artificialmente las métricas.

## 2. Universo elegible en el instante t

1. Crédito con **saldo positivo al cierre de `t`**.
2. `dias_mora(t)` exactamente **30 o 60**. Son buckets de **cuotas
   impagas equivalentes** y no días exactos desde el primer vencimiento.
3. **Tres cierres anteriores consecutivos** (`t−3`, `t−2`, `t−1`) con
   pago, cuota, saldo, fecha y mora no nulos; además de datos observados
   en la fecha de decisión. Las decisiones van de edad 4 a 9 meses.
4. **Tres cierres futuros consecutivos** (`t+1`, `t+2`, `t+3`) con mora
   observada y no nula: los huecos no se presuponen al día ni en mora.

No entran créditos sanos en `t`, ni aquellos en 90/120 DPD en `t`, ni
registros sin horizonte completo, sin saldo o con clave crédito-mes duplicada.
El estado de 120 es un bucket superior **no absorbente** en esta simulación:
una cuenta puede salir de él si abona el monto correspondiente. Una política
real requeriría definir castigos y otros estados terminales por separado.

## 3. Target binario y horizonte

El evento positivo es **curación contractual**:

`Y_i(t) = 1` si existe `k ∈ {1,2,3}` tal que `dias_mora_i(t+k) = 0`.

`Y_i(t) = 0` si ninguno de los tres cierres posteriores muestra cero DPD.

El horizonte es de **tres cierres mensuales** y comienza **después** de la
fecha de decisión. Una mejora de 60 a 30 DPD no es curación. Una cuenta que
cura en `t+1` y reincide en `t+3` se marca positiva: es curación *alguna
vez*, no curación sostenida. Este matiz debe discutirse en negocio.

La construcción utiliza `dias_mora(t+1..t+3)` **únicamente en la función
que crea la etiqueta**. Ningún campo posterior a `t` alimenta el modelo.

### Ejemplo temporal

| Mes | `t−2` | `t−1` | `t` (decisión) | `t+1` | `t+2` | `t+3` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mora | 0 | 30 | **60** | 60 | **0** | 30 |
| Uso | Observado | Observado | **Feature máxima** | Label | Label | Label |

Etiqueta: **1** porque se observa 0 DPD en `t+2`; no es condición que
permanezca al día al final del horizonte.

## 4. Censura, exclusiones e indeterminados

Si faltan uno o más cierres futuros **o si `dias_mora` es `NaN` en cualquiera
de ellos**, el préstamo es **no etiquetable** y se excluye; jamás se convierte
automáticamente en negativo. También se excluyen préstamos cuya historia
anterior carece de tres cierres reales consecutivos y completos.
En un panel real, prepago, venta de cartera, modificación, reestructuración,
castigo, fallecimiento o error operativo necesitarían reglas y posiblemente
análisis de riesgos competitivos. Aquí **no se modelan** esos fenómenos:
el panel sintético mantiene doce cierres por préstamo.

## 5. Qué significa valor capturado

`saldo_curado_i = saldo_i(t) × Y_i(t)`.

Este indicador es **exposición a la fecha de decisión en cuentas que
posteriormente llegaron a cero DPD**. NO es efectivo recuperado ni valor
contable realizado. La priorización por `p_i × saldo_i(t)` es una proxy
de valor esperado simple; en una operación real se incorporan monto de
cuotas vencidas, severidad, costos, probabilidad incremental de respuesta,
recovery cash-flow y restricciones de contacto.
