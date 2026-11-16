# Entorno congelado y procedencia del experimento

> **Datos sintéticos.** Estas versiones describen **el entorno que generó los
> reportes publicados**, no un entorno de una institución financiera.

## Entorno de referencia comprobado

| Componente | Versión utilizada |
| --- | --- |
| Python | **3.12.14** |
| Sistema | **Linux 6.18.57, x86_64, glibc 2.41** |
| NumPy | **2.3.5** |
| pandas | **2.2.3** |
| scikit-learn | **1.8.0** |
| Matplotlib | **3.10.8** |
| PyYAML | **6.0.3** |

`requirements.txt` fija **exactamente** las cinco dependencias directas con
`==`. El informe `reports/metricas.json` incorpora un bloque `entorno` con
la versión de Python, el sistema, la arquitectura, las versiones leídas en
tiempo de ejecución y el commit fuente. Si se modifica el entorno, el bloque
cambia y se hace visible la desviación.

En particular, **`HistGradientBoostingClassifier` puede cambiar su
comportamiento numérico entre versiones de scikit-learn**: la versión debe
coincidir para comparar métricas. También pueden influir NumPy, librerías
transitivas (SciPy, BLAS/OpenMP) y el orden de cómputo.

## ¿Qué identifica el hash del código?

`commit_codigo_fuente = 32705235` es la **revisión previamente auditada del
código de simulación, variables, estimación y evaluación** que produce estas
métricas. El cierre actual agrega fijación de dependencias, documentación y
metadatos **sin cambiar las operaciones predictivas**. No es el hash del
commit que publica este JSON. Introducir en un archivo el hash de **su propio
commit** crea una referencia circular: cualquier cambio al archivo altera
el SHA del commit. Por eso el hash se declara expresamente como **commit
fuente de código** y el commit de publicación se enlaza por separado.

## Verificación en una copia limpia

Usar preferiblemente Python **3.12.14** y el sistema Linux indicado:

```bash
git clone https://github.com/QuorumRisk/probabilidad-recuperacion-cartera.git
cd probabilidad-recuperacion-cartera
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

Las métricas/tablas son deterministas con la misma semilla y las versiones
probadas. **La garantía de comparación byte a byte se limita al entorno
efectivamente verificado**: un sistema operativo diferente, versiones
transitivas, fuentes o motor gráfico distintos pueden producir PNG diferentes
sin alterar las cifras. Se deben comprobar los hashes después de regenerar,
no inferir igualdad solo porque el comando termina sin errores.

**Verificación ejecutada en este cierre:** se creó una copia limpia del árbol
de Git del proyecto (`git clone --no-local`), se creó un `.venv` nuevo **sin
heredar las librerías del sistema**, se ejecutó `pip install -r requirements.txt`
y `python run_demo.py`, y se compararon los SHA-256 **antes y después**.
Coincidieron exactamente los **ocho artefactos versionados**: `metricas.json`,
`tabla_lift.csv`, `resumen.md` y los cinco PNG en `reports/figures/`.
Las **16 pruebas** terminaron correctamente también en la copia limpia.

Para comprobar los reportes antes y después de ejecutar, en la misma copia
puede usarse `sha256sum reports/metricas.json reports/tabla_lift.csv
reports/resumen.md reports/figures/*.png` y comparar los listados. Los
reportes son artefactos de la simulación, no resultados de una cartera real.
