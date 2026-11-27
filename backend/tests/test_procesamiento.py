"""Pruebas de los pesos 1/SE² por intervalo y de la prueba de sensibilidad a la rampa."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import engine  # noqa: E402
import main  # noqa: E402
import modelos_doctor as md  # noqa: E402

client = TestClient(main.app)
D = {"E": 20.0, "tau": 10.0, "alpha": 0.35}


def _curve(t):
    return engine.model_eval("relaxation", "FractionalMaxwellGel", md.to_native("relaxation", "FractionalMaxwellGel", D), t)


def _csv(**cols):
    names = list(cols)
    rows = [",".join(repr(float(cols[k][i])) for k in names) for i in range(len(cols[names[0]]))]
    return (",".join(names) + "\n" + "\n".join(rows)).encode()


def _ar1(rng, n, rho):
    e = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = e[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
    return out


def test_incertidumbre_por_intervalo_recupera_el_ruido_y_quita_la_tendencia():
    rng = np.random.default_rng(5)
    t = np.linspace(0.5, 600, 20000)
    y = _curve(t) * (1 + 0.005 * rng.standard_normal(len(t)))      # ruido relativo de 0.5 % por observación
    df = pd.DataFrame({"t": t, "y": y})
    se, sig, rho = main.bin_uncertainty(df, "t", ["y"])
    assert np.median(sig[:, 0]) == pytest.approx(0.005, rel=0.1)
    assert abs(rho[0]) < 0.05
    # sin ruido, la dispersión dentro de un intervalo es solo la tendencia de la curva: al quitarla queda ~0
    se0, sig0, _ = main.bin_uncertainty(pd.DataFrame({"t": t, "y": _curve(t)}), "t", ["y"])
    assert np.median(sig0[:, 0]) < 0.0005


def test_ruido_correlacionado_reduce_las_observaciones_efectivas():
    rng = np.random.default_rng(6)
    t = np.linspace(0.5, 600, 20000)
    y = _curve(t) * (1 + 0.005 * _ar1(rng, len(t), 0.8))
    se, sig, rho = main.bin_uncertainty(pd.DataFrame({"t": t, "y": y}), "t", ["y"])
    assert rho[0] == pytest.approx(0.8, abs=0.05)
    se_ind, _, _ = main.bin_uncertainty(pd.DataFrame({"t": t, "y": _curve(t) * (1 + 0.005 * rng.standard_normal(len(t)))}), "t", ["y"])
    # con rho = 0.8 el SE es sqrt((1 + rho)/(1 - rho)) = 3 veces mayor que con ruido independiente
    assert np.median(se[-20:, 0] / se_ind[-20:, 0]) == pytest.approx(3.0, rel=0.15)


def test_chi2_reducido_cercano_a_1_con_el_modelo_correcto():
    rng = np.random.default_rng(8)
    t = np.linspace(0.5, 600, 20000)
    y = _curve(t) * (1 + 0.005 * rng.standard_normal(len(t)))
    r = client.post("/api/fit", files={"file": ("s.csv", _csv(t=t, y=y))},
                    data={"test": "relaxation", "x_col": "t", "y_col": "y", "models": json.dumps(["FractionalMaxwellGel", "SpringPot"]),
                          "trim": "0", "weighting": "incertidumbre"})
    assert r.status_code == 200, r.text
    d = r.json()
    fmm = next(x for x in d["results"] if x["model"] == "FractionalMaxwellGel")
    sp = next(x for x in d["results"] if x["model"] == "SpringPot")
    assert 0.7 < fmm["chi2_red"] < 1.4          # el modelo que generó los datos los reproduce dentro del ruido
    assert sp["chi2_red"] > 10                    # el equivocado no
    P = fmm["doctor"]["parametros"]
    for k, v in D.items():
        assert P[k]["inf"] <= v <= P[k]["sup"]
    assert d["preprocesamiento"]["ponderacion"]["metodo"] == "incertidumbre"


def test_sensibilidad_a_la_rampa_consistente_con_el_modelo_exacto():
    rng = np.random.default_rng(9)
    t = np.linspace(0.5, 600, 6000)
    y = _curve(t) * (1 + 0.003 * rng.standard_normal(len(t)))
    data = pd.DataFrame({"t": t, "y": y})
    F = main._bin_and_fit("relaxation", data, "t", ["y"], "decada", ["FractionalMaxwellGel"])
    rec = engine.recommend(F["results"], len(F["x"]))
    S = main.ramp_sensitivity("relaxation", data, "t", ["y"], "decada", {"duracion_s": 1.0}, F["results"], rec)
    assert S["realizado"] and S["t_corte_s"] == 10.0
    m = S["modelos"]["FractionalMaxwellGel"]
    assert m["consistente"] is True, m
    assert set(m["parametros"]) == set(D)


def test_sensibilidad_sin_datos_suficientes():
    t = np.linspace(0.5, 30, 300)
    data = pd.DataFrame({"t": t, "y": _curve(t)})
    S = main.ramp_sensitivity("relaxation", data, "t", ["y"], "decada", {"duracion_s": 5.0}, [], None)
    assert S["realizado"] is False and "insuficientes" not in S  # 10·t1 = 50 s queda fuera de los datos


def test_ponderacion_incertidumbre_sin_promediar_usa_igual_peso():
    t = np.logspace(-1, 3, 80)
    r = client.post("/api/fit", files={"file": ("s.csv", _csv(t=t, y=_curve(t)))},
                    data={"test": "relaxation", "x_col": "t", "y_col": "y", "models": '["SpringPot"]', "trim": "0",
                          "weighting": "incertidumbre"})
    assert r.status_code == 200, r.text
    assert "aviso" in r.json()["preprocesamiento"]["ponderacion"]


def test_sensibilidad_detecta_un_sesgo_en_el_tramo_inicial():
    # un artefacto de +3 % que se desvanece en los primeros segundos (como una rampa mal corregida) debe detectarse
    rng = np.random.default_rng(10)
    t = np.linspace(0.5, 600, 6000)
    y = _curve(t) * (1 + 0.03 * np.exp(-t / 2.0)) * (1 + 0.003 * rng.standard_normal(len(t)))
    data = pd.DataFrame({"t": t, "y": y})
    F = main._bin_and_fit("relaxation", data, "t", ["y"], "decada", ["FractionalMaxwellGel"])
    S = main.ramp_sensitivity("relaxation", data, "t", ["y"], "decada", {"duracion_s": 1.0}, F["results"], None)
    assert S["modelos"]["FractionalMaxwellGel"]["consistente"] is False
