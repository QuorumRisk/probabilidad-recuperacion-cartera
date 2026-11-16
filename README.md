# Quórum Risk · Probabilidad de curación de mora

> ⚠️ **DEMOSTRACIÓN CON DATOS 100% SINTÉTICOS. NO SON DATOS DE CLIENTES, NI DE UN BANCO, NI DE UN PROVEEDOR EXTERNO.**
>
> Los gráficos y las métricas corresponden exclusivamente a una simulación
> reproducible. **No son evidencia de desempeño en carteras reales**, ni
> representan cifras comerciales de Quórum Risk.

## La conclusión primero: aquí NO recomendamos desplegar ML

En el experimento sintético, **ordenar las cuentas por saldo** captura
**41,09 %** de la exposición de cuentas que posteriormente curan al revisar
el 20 % de las cuentas. El mejor candidato ML por ese mismo criterio,
gradient boosting, alcanza **41,43 %: apenas +0,34 puntos porcentuales**.
La diferencia es pequeña, no se ha estimado su incertidumbre y no demuestra
ganancia económica ni un efecto causal de la gestión. **No hay evidencia
suficiente para recomendar pagar el costo y la complejidad de implantarlo**:
la regla sencilla es la recomendación prudente en esta simulación.

**Esto es criterio, no un fracaso del modelado.** Para Quórum Risk, un
*quórum* de enfoques exige que cada modelo justifique su entrada frente a
reglas operativas verificables: **si ML no aporta valor incremental convincente,
no entra en producción**. El objetivo del caso es mostrar cómo tomar esa
decisión con un target claro, control de fuga temporal y pruebas reproducibles.

**Evidencia:** [métricas](reports/metricas.json) ·
[comparación con reglas](reports/figures/valor_incremental.png) ·
[interpretación y límites](reports/resumen.md).

## El problema de negocio

Entre cuentas que **ya están en mora de 30–60 días**, ¿cuáles tienen mayor
probabilidad de **volver a estar al día dentro de los próximos tres meses**?
Un score puede ayudar a organizar una cola de análisis o priorización de
cobranza. Su valor depende de definir el evento correctamente, congelar
información disponible al decidir y evaluar en una cosecha posterior.

Este caso hace visible lo que sucede **antes** del entrenamiento: universo,
target temporal, variables explicables, anti-fuga y criterios de evaluación.
No pretende demostrar que enviar más llamadas genere más curaciones.

## Resultado reproducible

La ejecución genera tres candidatos: **regresión logística**, **gradient
boosting** y una **combinación ponderada**. El peso se escoge únicamente con
validación 2021; el test 2022 no participa en esa decisión. Si el modelo
sencillo supera a la combinación, se reporta sin maquillarlo.

| Corte de información | Horizonte | Validación | Variable de negocio |
| --- | --- | --- | --- |
| Cierre mensual con mora 30/60 DPD | `t+1` a `t+3` | Cosechas 2019–20 → 2021 → **2022** | Saldo de cuentas que posteriormente curan |

**Advertencia de interpretación:** la curva de "valor recuperable" utiliza el
**saldo vigente en `t` de créditos que después curaron**. No mide pesos/dólares
efectivamente cobrados, ganancia incremental, costos de gestión ni
probabilidad de recuperar el 100% de la exposición. El ranking ilustrativo
ordena por `P(curación) × saldo_en_t`.

Resultados generados con la semilla pública:

- [Resumen ejecutivo técnico](reports/resumen.md)
- [Métricas por modelo](reports/metricas.json)
- [Lift por decil](reports/tabla_lift.csv)
- [Comparación con reglas sin ML](reports/figures/valor_incremental.png)

![Concentración del saldo en cuentas que curan — simulación](reports/figures/valor_capturado.png)

También: [ROC y precisión-recall](reports/figures/discriminacion.png) ·
[Calibración](reports/figures/calibracion.png) · [Lift](reports/figures/lift.png).

## Reproducir desde cero

**Entorno de referencia: Python 3.12.14 y dependencias exactas** declaradas
en `requirements.txt`. Consulta la [ficha de entorno y procedencia](docs/ENTORNO.md)
para el sistema Linux probado, las versiones y el significado del commit fuente.
**Los resultados reproducen bit a bit instalando las versiones fijadas en el
entorno verificado**: en una copia limpia con entorno virtual aislado se
obtuvieron hashes SHA-256 idénticos para `metricas.json`, `tabla_lift.csv`,
`resumen.md` y las cinco figuras. No extendemos esa garantía a otros sistemas
operativos, motores gráficos o fuentes sin volver a comprobarlos.
Desde la raíz del repositorio:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` crea `data/` automáticamente y escribe allí `panel_sintetico.csv`
y `decisiones_sinteticas.csv` para inspección local. **`data/` está ignorado
por Git y ningún dato crudo se publica**. Los archivos versionados bajo
`reports/` son agregados, figuras y resúmenes de una simulación.

Semilla: `20261116` · **2.600 créditos originados el 1 de enero de cada
cosecha (2019, 2020, 2021 y 2022)** · **12 cierres mensuales por préstamo**. Cambiar la semilla
o el tamaño altera los resultados:

```bash
python run_demo.py --semilla 20261116 --prestamos-por-cosecha 2600
```

## La metodología, como estándar Quórum Risk

| Etapa | Documento / módulo | Qué protege |
| --- | --- | --- |
| Universo y target | [`docs/DEFINICION_TARGET.md`](docs/DEFINICION_TARGET.md) | Qué se intenta anticipar, cuándo y para quién |
| Panel y reglas sintéticas | [`src/data/generador.py`](src/data/generador.py) | Trazabilidad del simulador, sin datos privados |
| Preparación | [`src/data/preparacion.py`](src/data/preparacion.py) | Horizonte completo, una foto por cuenta |
| Variables | [`docs/FEATURE_ENGINEERING.md`](docs/FEATURE_ENGINEERING.md) | Variables con sentido de negocio, solo datos hasta `t` |
| Validación | [`docs/VALIDACION.md`](docs/VALIDACION.md) | Separación estricta de cosechas y métricas |
| Candidatos | [`src/models/`](src/models/) | Código, hiperparámetros y explicación propios |
| Evaluación | [`src/evaluacion/`](src/evaluacion/) | Calibración, discriminación y concentración de saldos |
| Calidad | [`tests/`](tests/) | Invariancia temporal, reproducibilidad y agregaciones |

### Reglas del generador sintético

1. Los préstamos nacen en **enero de 2019, 2020, 2021 o 2022**. Sus atributos
   iniciales (ingreso, DTI, grado, propósito, saldo inicial, plazo y tasa)
   proceden de distribuciones documentadas en `src/data/generador.py`.
2. En cada cierre se fija la **cuota contractual** y se construye el libro
   monetario de vencidos: **exigible = atraso inicial + cuota del mes**;
   **atraso al cierre = máximo(0, exigible − pago realizado)**. Se guardan
   explícitamente `cuota_programada`, `pago_realizado`,
   `saldo_vencido_inicio`, `obligacion_exigible` y `saldo_vencido_cierre`.
3. **El DPD no se sortea ni salta arbitrariamente**: se deriva de las
   cuotas vencidas equivalentes al cierre, usando tramos de 30 días
   (`0 / 30 / 60 / 90 / 120`, con 120 como límite superior). Una cuenta
   únicamente se cura cuando paga **atraso previo y cuota actual**. Con
   pago menor que la cuota puede deteriorarse; si paga al menos la cuota,
   no puede aumentar su atraso monetario. `120` **no es absorbente**.
4. La aleatoriedad actúa **solo en cuánto paga cada cliente**: la
   probabilidad y magnitud del pago dependen de grado, carga de cuota,
   capacidad latente, shock mensual y saldo ya vencido; 2022 incorpora
   tensión ambiental ficticia. El principal/los intereses se actualizan
   por una regla simplificada, **no un motor contable real**. Los buckets
   de DPD son **equivalentes de cuotas impagas**, no la antigüedad exacta
   de la factura vencida más antigua de un sistema de servicing.
5. El azar evita que la curación sea perfectamente predecible. No se crean
   etiquetas artificiales independientes del historial: el target se obtiene
   **leyendo los estados futuros reales dentro de la propia simulación**.
6. Es un mecanismo **inventado para enseñar el pipeline**, no calibrado con
   estadísticas históricas de ninguna institución. No se debe interpretar
   como una estimación de la mora o la curación de una población real.

### Anti-fuga: variables permitidas y excluidas

- **Disponibles:** características del contrato en originación, saldo
  **al cierre de `t`**, saldo vencido de `t`, DPD de `t`, pago observado hasta `t`, tasas/ratios,
  recencia y tendencias calculadas con hasta seis meses previos.
- **Excluidos de la matriz del modelo:** `loan_id`, `cosecha`,
  `fecha_decision`, `label_cura_3m`, `saldo_cura_observado`; todos los
  `dias_mora`, `pago_realizado` o `saldo_actual` de meses `t+1` a `t+3`;
  y los shocks y la resiliencia latente del generador. El historial futuro
  **solo** se usa para etiquetar/evaluar, nunca para construir features.
- Una lista positiva `COLUMNAS_MODELO` garantiza el esquema. Un test cambia
  deliberadamente los estados futuros y verifica que **las features no cambien**.
- Una decisión exige **tres cierres anteriores consecutivos** (`t−3` a
  `t−1`, con datos observados) y **tres posteriores consecutivos con DPD
  conocido** (`t+1` a `t+3`). Un faltante o `NaN` futuro vuelve la
  observación **no etiquetable**: nunca se sustituye por etiqueta negativa.

### ¿Aporta algo frente a una regla sencilla?

Antes de justificar un score se comparan dos políticas **sin ML**:

1. Atender primero los créditos de **mayor saldo** al cierre `t`.
2. Atender primero las cuentas en **30 DPD**, después las de 60 DPD,
   desempatando por saldo descendente. Es una regla ilustrativa, no una
   política validada en un negocio real.

La métrica común es el **porcentaje de exposición en cuentas que curaron**
que se concentra al priorizar el **20% de cuentas**. En el informe se
contrasta cada candidato con la **mejor de estas dos reglas** en puntos
porcentuales; una diferencia observada **no es causalidad, cobro realizado
ni beneficio económico**. [Ver medición regenerada](reports/resumen.md).

## Alcance y fuentes

No se ha utilizado LendingClub, Freddie Mac, Fannie Mae ni ninguna otra
base con restricciones comerciales: los datos nacen del generador original
de este proyecto. No hay descarga, redistribución ni transformación de
información de terceros. El trabajo presentado es demostrativo y el código
es propio de Quórum Risk.

En un encargo real el target se acuerda con Cobranza y Riesgo, se auditan
calidad/tiempos de disponibilidad, se revisan sesgos y regulación, se
monitoriza estabilidad y calibración, y **los datos del cliente no se
comparten ni publican**.

**Contacto:** [contacto@quorumrisk.co](mailto:contacto@quorumrisk.co) ·
[quorumrisk.co](https://quorumrisk.co)
