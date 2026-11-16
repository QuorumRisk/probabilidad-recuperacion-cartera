"""Clasificador interpretable de referencia."""

from pathlib import Path

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.construccion import CATEGORICAS, NUMERICAS


def crear_modelo() -> Pipeline:
    ruta = Path(__file__).with_name("hiperparametros.yaml")
    parametros = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    numericas = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())])
    categoricas = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preparacion = ColumnTransformer(
        [("num", numericas, NUMERICAS), ("cat", categoricas, CATEGORICAS)],
        remainder="drop",
        sparse_threshold=0,
    )
    return Pipeline([("preparacion", preparacion), ("clasificador", LogisticRegression(**parametros))])
