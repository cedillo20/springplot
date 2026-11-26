"""Pruebas de los modelos hiperelásticos (Neo-Hookeano y Mooney-Rivlin) en tensión uniaxial.

Correr desde la carpeta backend:  python -m pytest tests -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import hiperelastico as he  # noqa: E402
import main  # noqa: E402
import tensile  # noqa: E402

client = TestClient(main.app)


def _csv(**cols):
    names = list(cols)
    rows = [",".join(repr(float(cols[k][i])) for k in names) for i in range(len(cols[names[0]]))]
    return (",".join(names) + "\n" + "\n".join(rows)).encode()


def test_ecuaciones_de_tension_uniaxial():
    lam = np.array([1.5, 2.0])
    e = (lam - 1) * 100
    assert he.stress("NeoHookeano", {"C10": 0.5}, e) == pytest.approx(2 * 0.5 * (lam - lam ** -2))
    assert he.stress("MooneyRivlin", {"C10": 0.3, "C01": 0.2}, e) == pytest.approx(2 * (0.3 + 0.2 / lam) * (lam - lam ** -2))
    # deformación pequeña: pendiente inicial E0 = 6 (C10 + C01) = 3 mu
    small = he.stress("MooneyRivlin", {"C10": 0.3, "C01": 0.2}, [0.01])[0] / 1e-4
    assert small == pytest.approx(6 * 0.5, rel=1e-3)


@pytest.mark.parametrize("model,p", [("NeoHookeano", {"C10": 0.6}), ("MooneyRivlin", {"C10": 0.4, "C01": 0.25})])
def test_recupera_constantes_con_ruido(model, p):
    rng = np.random.default_rng(4)
    e = np.linspace(0.5, 300, 600)
    s = he.stress(model, p, e) * (1 + 0.005 * rng.standard_normal(len(e)))
    r = he.fit(e, s, len(e) - 1)
    m = r["modelos"][model]
    for k, v in p.items():
        assert m["ic95"][k]["inf"] <= v <= m["ic95"][k]["sup"]
        assert m["parametros"][k] == pytest.approx(v, rel=0.02)
    assert m["mu"] == pytest.approx(2 * sum(p.values()), rel=0.01)
    assert r["variacion_mu_pct"][model] < 5        # el modelo correcto no depende del rango ajustado
    if model == "MooneyRivlin":
        assert r["modelos"]["NeoHookeano"]["aad_pct"] > m["aad_pct"]


def test_grafica_de_mooney_es_recta_para_mooney_rivlin():
    e = np.linspace(0.5, 200, 400)
    r = he.fit(e, he.stress("MooneyRivlin", {"C10": 0.4, "C01": 0.25}, e), len(e) - 1)
    x, y = np.array(r["mooney"]["inv_lambda"]), np.array(r["mooney"]["esfuerzo_reducido"])
    slope, icpt = np.polyfit(x, y, 1)
    assert icpt == pytest.approx(0.4, rel=1e-6) and slope == pytest.approx(0.25, rel=1e-6)


def test_limite_de_deformacion_y_aviso_de_rango_corto():
    e = np.linspace(0.5, 300, 600)
    s = he.stress("MooneyRivlin", {"C10": 0.4, "C01": 0.25}, e)
    r = he.fit(e, s, len(e) - 1, max_pct=10)
    assert r["rango_deformacion_pct"][1] <= 10 and r["fin_rama_carga_pct"] == pytest.approx(300)
    assert "aviso_rango" in r


def test_avisos_de_estabilidad_y_correlacion():
    e = np.linspace(0.5, 50, 300)
    s = he.stress("MooneyRivlin", {"C10": -0.5, "C01": 1.5}, e)
    m = he.fit(e, s, len(e) - 1)["modelos"]["MooneyRivlin"]
    assert any("Drucker" in a for a in m["avisos"])


def test_tension_d638_incluye_hiperelastico_y_valida_el_limite():
    # curva de Mooney-Rivlin con pie de curva corto, sin ruptura
    e = np.linspace(0, 150, 1500)
    s = he.stress("MooneyRivlin", {"C10": 0.4, "C01": 0.25}, e)
    r = client.post("/api/tensile", files={"file": ("s.csv", _csv(e=e, s=s))}, data={"x_col": "e", "y_col": "s", "secant": "1"})
    assert r.status_code == 200, r.text
    d = r.json()
    H = d["results"][0]["hiperelastico"]
    assert H["modelos"]["MooneyRivlin"]["parametros"]["C01"] == pytest.approx(0.25, rel=0.02)
    assert d["hiperelastico_resumen"]["MooneyRivlin"]["mu"]["n"] == 1
    assert any(c["clausula"] == "Modelos hiperelásticos" for c in d["verificaciones"])
    bad = client.post("/api/tensile", files={"file": ("s.csv", _csv(e=e, s=s))},
                      data={"x_col": "e", "y_col": "s", "secant": "1", "hyper_max": "-5"})
    assert bad.status_code == 400
