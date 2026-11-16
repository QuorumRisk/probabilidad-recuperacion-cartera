# El entorno con el que se generaron los resultados

Los datos de este caso son sintéticos. Las versiones que siguen corresponden a la ejecución que produjo los informes publicados, no al entorno de un banco o de un cliente.

## Versiones utilizadas

La ejecución se hizo con Python 3.12.14 sobre Linux 6.18.57, arquitectura x86_64 y glibc 2.41.

Las bibliotecas fueron NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, Matplotlib 3.10.8 y PyYAML 6.0.3. Las cinco están fijadas con `==` en `requirements.txt`.

El archivo `reports/metricas.json` guarda estas versiones dentro del bloque `entorno`, junto con Python, sistema, arquitectura y commit fuente. Si alguien ejecuta con versiones diferentes, el bloque permite ver esa diferencia.

La versión de scikit-learn importa especialmente para `HistGradientBoostingClassifier`, cuyo comportamiento numérico puede cambiar entre versiones. NumPy, SciPy, BLAS/OpenMP y el orden del cálculo también pueden influir.

## A qué commit corresponde el experimento

`commit_codigo_fuente = 32705235` identifica la revisión auditada del generador, las variables, los modelos y la evaluación que produjo las métricas. El cierre posterior añadió dependencias fijadas, documentación y metadatos sin cambiar las operaciones predictivas.

No es el hash del commit que contiene el JSON. Guardar dentro de un archivo el hash de su propio commit genera una referencia circular, porque editar el archivo cambia ese hash. Por eso distinguimos el commit del código fuente y el commit que publica los informes.

## Cómo repetir la ejecución

Recomendamos usar Python 3.12.14 en el Linux indicado. Desde una terminal:

```bash
git clone https://github.com/QuorumRisk/probabilidad-recuperacion-cartera.git
cd probabilidad-recuperacion-cartera
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

En la verificación de este cierre se utilizó una copia limpia del árbol de Git creada con `git clone --no-local`. Se instaló un `.venv` nuevo, sin heredar librerías del sistema, y se compararon los SHA-256 antes y después de ejecutar. Los ocho artefactos coincidieron: `metricas.json`, `tabla_lift.csv`, `resumen.md` y los cinco PNG de `reports/figures/`. También pasaron las 16 pruebas.

Esa igualdad byte a byte se comprobó en el entorno descrito. Un sistema operativo distinto, otras versiones transitivas o un motor gráfico con fuentes diferentes pueden generar PNG distintos sin cambiar las métricas. Conviene comprobarlo, no darlo por hecho.

Para comparar los archivos se puede ejecutar:

```bash
sha256sum reports/metricas.json reports/tabla_lift.csv reports/resumen.md reports/figures/*.png
```

Los reportes corresponden a una simulación. Que se puedan reproducir no significa que describan el comportamiento de una cartera real.
