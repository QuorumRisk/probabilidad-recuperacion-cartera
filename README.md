# Quórum Risk · Probabilidad de curación de mora

> ⚠️ **DEMOSTRACIÓN CON DATOS 100% SINTÉTICOS. NO SON DATOS DE CLIENTES, NI DE UN BANCO, NI DE UN PROVEEDOR EXTERNO.**
>
> Los gráficos y las métricas corresponden exclusivamente a una simulación
> reproducible. **No son evidencia de desempeño en carteras reales**, ni
> representan cifras comerciales de Quórum Risk.

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

![Concentración del saldo en cuentas que curan — simulación](reports/figures/valor_capturado.png)

También: [ROC y precisión-recall](reports/figures/discriminacion.png) ·
[Calibración](reports/figures/calibracion.png) · [Lift](reports/figures/lift.png).

## Reproducir desde cero

Necesitas Python 3.10+ (recomendado 3.11–3.12). Desde la raíz del repositorio:

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

Semilla: `20261116` · **2.600 créditos originados en enero por cada una de
cuatro cosechas** · **12 cierres mensuales por préstamo**. Cambiar la semilla
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
2. En cada cierre mensual se simula el **pago observado**, con ruido y un
   componente de resiliencia no observable. El saldo se actualiza según pago
   y devengo simplificados; **no es un motor contable**.
3. El estado de mora recorre `0 → 30 → 60 → 90 → 120` días y admite curaciones,
   mejoras y deterioros. La probabilidad de transición depende de grado,
   carga financiera, pagos, shock mensual y un componente oculto. El estado
   `120` es absorbente en este ejemplo. El año 2022 incorpora una tensión de
   entorno ilustrativa.
4. El azar evita que la curación sea perfectamente predecible. No se crean
   etiquetas artificiales independientes del historial: el target se obtiene
   **leyendo los estados futuros reales dentro de la propia simulación**.
5. Es un mecanismo **inventado para enseñar el pipeline**, no calibrado con
   estadísticas históricas de ninguna institución. No se debe interpretar
   como una estimación de la mora o la curación de una población real.

### Anti-fuga: variables permitidas y excluidas

- **Disponibles:** características del contrato en originación, saldo
  **al cierre de `t`**, DPD de `t`, pago observado hasta `t`, tasas/ratios,
  recencia y tendencias calculadas con hasta seis meses previos.
- **Excluidos de la matriz del modelo:** `loan_id`, `cosecha`,
  `fecha_decision`, `label_cura_3m`, `saldo_cura_observado`; todos los
  `dias_mora`, `pago_realizado` o `saldo_actual` de meses `t+1` a `t+3`;
  y los shocks y la resiliencia latente del generador. El historial futuro
  **solo** se usa para etiquetar/evaluar, nunca para construir features.
- Una lista positiva `COLUMNAS_MODELO` garantiza el esquema. Un test cambia
  deliberadamente los estados futuros y verifica que **las features no cambien**.

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
