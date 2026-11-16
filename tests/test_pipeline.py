"""Pruebas de integridad temporal, semilla y métricas, sin librerías extra."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.generador import generar_panel
from src.data.preparacion import definir_observaciones
from src.evaluacion.metricas import calcular_metricas, curva_valor_capturado, tabla_lift
from src.features.construccion import COLUMNAS_MODELO, construir_variables
from src.models.combinacion.modelo import combinar, seleccionar_peso


class PruebasDelCaso(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = generar_panel(semilla=20261116, prestamos_por_cosecha=130)
        cls.decisiones = definir_observaciones(cls.panel)
        cls.variables = construir_variables(cls.panel, cls.decisiones)

    def test_semilla_regenera_exactamente_el_mismo_panel(self):
        otro = generar_panel(semilla=20261116, prestamos_por_cosecha=130)
        pd.testing.assert_frame_equal(self.panel, otro, check_exact=True)

    def test_universo_y_etiqueta_binaria(self):
        d = self.decisiones
        self.assertFalse(d["loan_id"].duplicated().any())
        self.assertTrue(d["dias_mora"].isin([30, 60]).all())
        self.assertTrue(d["mes_edad"].between(4, 9).all())
        self.assertTrue(d["saldo_decision"].gt(0).all())
        self.assertTrue(d["label_cura_3m"].isin([0, 1]).all())

    def test_features_invariantes_a_perturbar_futuro(self):
        alterado = self.panel.merge(
            self.decisiones[["loan_id", "mes_edad"]].rename(columns={"mes_edad": "mes_decision"}),
            on="loan_id", how="left",
        )
        mask = alterado["mes_edad"].gt(alterado["mes_decision"])
        alterado.loc[mask, "dias_mora"] = 0
        alterado.loc[mask, "pago_realizado"] = 999999.0
        alterado.loc[mask, "saldo_actual"] = 999999.0
        alterado = alterado.drop(columns="mes_decision")
        otros_predictores = construir_variables(alterado, self.decisiones)
        pd.testing.assert_frame_equal(
            self.variables.sort_values("loan_id")[COLUMNAS_MODELO].reset_index(drop=True),
            otros_predictores.sort_values("loan_id")[COLUMNAS_MODELO].reset_index(drop=True),
            check_exact=True,
        )

    def test_target_reacciona_a_futuro_sin_mover_el_corte(self):
        negativos = self.decisiones.loc[self.decisiones["label_cura_3m"].eq(0)]
        self.assertGreater(len(negativos), 0)
        ejemplo = negativos.iloc[0]
        p = self.panel.copy()
        mask = (p["loan_id"] == ejemplo["loan_id"]) & (
            p["mes_edad"].between(int(ejemplo["mes_edad"]) + 1, int(ejemplo["mes_edad"]) + 3)
        )
        p.loc[mask, "dias_mora"] = 0
        etiquetas_nuevas = definir_observaciones(p).set_index("loan_id")
        self.assertEqual(int(etiquetas_nuevas.loc[ejemplo["loan_id"], "label_cura_3m"]), 1)

    def test_cosechas_sin_cruce_cronologico(self):
        d = self.decisiones
        for anio in (2019, 2020, 2021, 2022):
            cohorte = d[d["cosecha"].eq(anio)]
            self.assertGreater(len(cohorte), 0)
            self.assertTrue(cohorte["fecha_decision"].dt.year.eq(anio).all())
            self.assertTrue(cohorte["fecha_decision"].dt.month.le(9).all())
        self.assertLess(d.loc[d["cosecha"].eq(2021), "fecha_decision"].max() + pd.DateOffset(months=3),
                        d.loc[d["cosecha"].eq(2022), "fecha_decision"].min())

    def test_modelos_no_reciben_target_ni_metadatos(self):
        for prohibida in ["loan_id", "cosecha", "fecha_decision", "label_cura_3m", "saldo_cura_observado"]:
            self.assertNotIn(prohibida, COLUMNAS_MODELO)
        self.assertEqual(len(COLUMNAS_MODELO), len(set(COLUMNAS_MODELO)))
        self.assertFalse(self.variables[COLUMNAS_MODELO].isna().any().any())

    def test_combinacion_y_metricas(self):
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        p1 = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.1, 0.8, 0.2, 0.7, 0.2])
        p2 = np.array([0.3, 0.7, 0.2, 0.6, 0.8, 0.2, 0.7, 0.4, 0.8, 0.3])
        peso, briers = seleccionar_peso(y, p1, p2)
        self.assertIn(peso, [0.25, 0.5, 0.75])
        self.assertEqual(len(briers), 3)
        mezcla = combinar(p1, p2, peso)
        self.assertTrue(np.all((mezcla >= 0) & (mezcla <= 1)))
        m = calcular_metricas(y, mezcla)
        self.assertGreater(m["auc_roc"], 0.5)
        saldos = np.linspace(2000, 12000, len(y))
        curva = curva_valor_capturado(y, mezcla, saldos)
        self.assertAlmostEqual(float(curva["saldo_curado_capturado_pct"].iloc[-1]), 100.0)
        self.assertTrue(curva["saldo_curado_capturado_pct"].is_monotonic_increasing)
        self.assertEqual(int(tabla_lift(y, mezcla, saldos, "prueba")["cuentas"].sum()), len(y))


if __name__ == "__main__":
    unittest.main()
