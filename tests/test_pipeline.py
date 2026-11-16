"""Pruebas de integridad temporal, semilla y métricas, sin librerías extra."""

from __future__ import annotations

import json
import platform
import unittest
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd

from src.data.generador import generar_panel
from src.data.preparacion import definir_observaciones
from src.evaluacion.metricas import (
    calcular_metricas, captura_top_20, curva_valor_capturado,
    prioridades_sin_ml, tabla_lift,
)
from src.features.construccion import COLUMNAS_MODELO, construir_variables
from src.models.combinacion.modelo import combinar, seleccionar_peso
from run_demo import COMMIT_CODIGO_FUENTE


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

    def test_sin_historia_de_tres_cierres_no_hay_decision(self):
        ejemplo = self.decisiones.iloc[0]
        prestamo = ejemplo["loan_id"]
        t = int(ejemplo["mes_edad"])
        panel = self.panel.loc[
            ~((self.panel["loan_id"] == prestamo) & (self.panel["mes_edad"] < t))
        ].copy()
        resultado = definir_observaciones(panel)
        self.assertNotIn(prestamo, set(resultado["loan_id"]))

    def test_una_laguna_en_historia_no_cuenta_como_cierre(self):
        ejemplo = self.decisiones.iloc[1]
        prestamo = ejemplo["loan_id"]
        t = int(ejemplo["mes_edad"])
        panel = self.panel.loc[
            ~((self.panel["loan_id"] == prestamo) & (self.panel["mes_edad"] == t - 2))
        ].copy()
        resultado = definir_observaciones(panel)
        self.assertNotIn(prestamo, set(resultado["loan_id"]))

    def test_dpd_futuro_nan_no_se_etiqueta_negativo(self):
        ejemplo = self.decisiones.iloc[2]
        prestamo = ejemplo["loan_id"]
        panel = self.panel.copy()
        mask = panel["loan_id"].eq(prestamo) & panel["mes_edad"].eq(int(ejemplo["mes_edad"]) + 2)
        panel.loc[mask, "dias_mora"] = np.nan
        self.assertEqual(int(mask.sum()), 1)
        resultado = definir_observaciones(panel)
        self.assertNotIn(prestamo, set(resultado["loan_id"]))

    def test_falta_cierre_futuro_no_se_etiqueta(self):
        ejemplo = self.decisiones.iloc[3]
        prestamo = ejemplo["loan_id"]
        panel = self.panel.loc[
            ~((self.panel["loan_id"] == prestamo) &
              self.panel["mes_edad"].eq(int(ejemplo["mes_edad"]) + 1))
        ].copy()
        resultado = definir_observaciones(panel)
        self.assertNotIn(prestamo, set(resultado["loan_id"]))

    def test_mora_se_deriva_del_libro_de_cuotas(self):
        p = self.panel
        exigible = np.round(p["saldo_vencido_inicio"] + p["cuota_programada"], 2)
        cierre = np.round(np.maximum(0.0, exigible - p["pago_realizado"]), 2)
        np.testing.assert_allclose(p["obligacion_exigible"], exigible, atol=0.011)
        np.testing.assert_allclose(p["saldo_vencido_cierre"], cierre, atol=0.011)
        mora_esperada = np.minimum(
            np.ceil(np.maximum(0.0, cierre / p["cuota_programada"] - 1e-10)).astype(int), 4
        ) * 30
        np.testing.assert_array_equal(p["dias_mora"], mora_esperada)

    def test_no_hay_curas_sin_pagar_todo_lo_exigible(self):
        p = self.panel.sort_values(["loan_id", "mes_edad"]).copy()
        previo = p.groupby("loan_id")["dias_mora"].shift()
        cura = previo.gt(0) & p["dias_mora"].eq(0)
        self.assertGreater(int(cura.sum()), 0)
        self.assertTrue(
            (p.loc[cura, "pago_realizado"] >= p.loc[cura, "obligacion_exigible"] - 0.011).all()
        )
        entra = previo.eq(0) & p["dias_mora"].gt(0)
        self.assertGreater(int(entra.sum()), 0)
        self.assertTrue(
            (p.loc[entra, "pago_realizado"] < p.loc[entra, "cuota_programada"]).all()
        )
        empeora = previo.notna() & p["dias_mora"].gt(previo)
        self.assertTrue(
            (p.loc[empeora, "pago_realizado"] < p.loc[empeora, "cuota_programada"]).all()
        )

    def test_reglas_baseline_priorizan_sin_conocer_el_target(self):
        y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 0])
        saldos = np.array([1, 10, 8, 9, 7, 6, 5, 4, 3, 2], dtype=float)
        dpd = np.array([30, 60, 30, 60, 30, 60, 30, 60, 30, 60])
        baselines = prioridades_sin_ml(saldos, dpd)
        self.assertEqual(
            np.argsort(-baselines["saldo_descendente"], kind="stable")[:2].tolist(), [1, 3]
        )
        self.assertEqual(
            np.argsort(-baselines["dpd_30_antes_60_desempate_saldo"], kind="stable")[:2].tolist(), [2, 4]
        )
        total = float(np.dot(y, saldos))
        captura_saldo = captura_top_20(y, np.zeros(len(y)), saldos,
                                      prioridad=baselines["saldo_descendente"])
        captura_dpd = captura_top_20(y, np.zeros(len(y)), saldos,
                                    prioridad=baselines["dpd_30_antes_60_desempate_saldo"])
        self.assertAlmostEqual(captura_saldo, round(100 * 10 / total, 6))
        self.assertAlmostEqual(captura_dpd, round(100 * 15 / total, 6))

    def test_resumen_publicado_concuerda_con_metricas_json(self):
        raiz = Path(__file__).resolve().parents[1] / "reports"
        datos = json.loads((raiz / "metricas.json").read_text(encoding="utf-8"))
        resumen = (raiz / "resumen.md").read_text(encoding="utf-8")
        for nombre, modelo in datos["modelos"].items():
            valor = modelo["captura_saldo_curado_top_20_pct"]
            diferencial = modelo["uplift_vs_mejor_baseline_pp"]
            self.assertIn(f"| {nombre} |", resumen)
            self.assertIn(f"{valor:.2f}%", resumen)
            self.assertIn(f"{diferencial:+.2f} pp", resumen)
        for base in datos["baselines"].values():
            self.assertIn(f"{base['captura_saldo_curado_top_20_pct']:.2f}%", resumen)
        mejor = datos["mejor_baseline"]
        valor_referencia = datos["baselines"][mejor]["captura_saldo_curado_top_20_pct"]
        for modelo in datos["modelos"].values():
            self.assertAlmostEqual(
                modelo["uplift_vs_mejor_baseline_pp"],
                modelo["captura_saldo_curado_top_20_pct"] - valor_referencia,
                places=5,
            )

    def test_entorno_publicado_coincide_con_dependencias_fijadas(self):
        raiz = Path(__file__).resolve().parents[1]
        datos = json.loads((raiz / "reports" / "metricas.json").read_text(encoding="utf-8"))
        entorno = datos["entorno"]
        self.assertEqual(entorno["python"], platform.python_version())
        self.assertEqual(entorno["sistema"], platform.system())
        self.assertEqual(entorno["commit_codigo_fuente"], COMMIT_CODIGO_FUENTE)
        for linea in (raiz / "requirements.txt").read_text(encoding="utf-8").splitlines():
            nombre, version_fijada = linea.strip().split("==")
            self.assertEqual(version_fijada, version(nombre))
            self.assertEqual(entorno["librerias"][nombre], version_fijada)


if __name__ == "__main__":
    unittest.main()
