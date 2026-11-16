"""Modelo no lineal para interacciones entre historia y perfil."""

from pathlib import Path

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.construccion import CATEGORICAS, NUMERICAS


def crear_modelo() -> Pipeline:
    ruta = Path(__file__).with_name("hiperparametros.yaml")
    parametros = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    numericas = SimpleImputer(strategy="median")
    categoricas = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preparacion = ColumnTransformer(
        [("num", numericas, NUMERICAS), ("cat", categoricas, CATEGORICAS)],
        remainder="drop",
        sparse_threshold=0,
    )
    return Pipeline([("preparacion", preparacion), ("clasificador", HistGradientBoostingClassifier(**parametros))])
