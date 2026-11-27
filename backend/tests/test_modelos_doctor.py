"""Pruebas de Springpot, FMM, KVF y ZFM en notación del doctor, en creep y relajación.
La prueba de cobertura usa 40 réplicas; con SPRINGPLOT_SLOW=1 usa 200.
"""
import json
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import engine  # noqa: E402
import modelos_doctor as md  # noqa: E402
import main  # noqa: E402
import varpro  # noqa: E402

client = TestClient(main.app)
T = np.logspace(-1, 3, 120)


def _csv(**cols):
    names = list(cols)
    rows = [",".join(repr(float(cols[k][i])) for k in names) for i in range(len(cols[names[0]]))]
    return (",".join(names) + "\n" + "\n".join(rows)).encode()


def _fit_api(test, t, y, models, **extra):
    r = client.post("/api/fit", files={"file": ("s.csv", _csv(t=t, y=y))},
                    data={"test": test, "x_col": "t", "y_col": "y", "models": json.dumps(models), "trim": "0", **extra})
    assert r.status_code == 200, r.text
    return r.json()


def _eval(test, model, d, t):
    return engine.model_eval(test, model, md.to_native(test, model, d), np.asarray(t, float))


# parámetros con el tiempo característico dentro de la ventana
CASES = [
    ("creep", "FractionalMaxwellGel", {"E": 20.0, "tau": 10.0, "alpha": 0.35}),
    ("relaxation", "FractionalMaxwellGel", {"E": 20.0, "tau": 10.0, "alpha": 0.35}),
    ("creep", "FractionalKelvinVoigtS", {"E": 20.0, "tau": 10.0, "alpha": 0.35}),
    ("relaxation", "FractionalKelvinVoigtS", {"E": 20.0, "tau": 10.0, "alpha": 0.35}),
    ("creep", "FractionalZenerSolidS", {"E_r": 10.0, "E_u": 30.0, "tau_epsilon": 20.0, "alpha": 0.4}),
    ("relaxation", "FractionalZenerSolidS", {"E_r": 10.0, "E_u": 30.0, "tau_sigma": 5.0, "alpha": 0.4}),
]


# Recuperación de parámetros conocidos
@pytest.mark.parametrize("test,model,d", CASES)
def test_recupera_parametros_del_doctor_con_ruido_de_1pct(test, model, d):
    rng = np.random.default_rng(7)
    y = _eval(test, model, d, T) * (1 + 0.01 * rng.standard_normal(len(T)))
    res = _fit_api(test, T, y, [model])["results"][0]
    assert "error" not in res, res
    P = res["doctor"]["parametros"]
    # tolerancia: 2 % con 3 parámetros, 3 % con 4
    tol = 0.02 if len(d) == 3 else 0.03
    for k, v in d.items():
        u = P[k]
        assert u["determinado"], (k, u)
        assert u["inf"] <= v <= u["sup"], (k, v, u)          # valor verdadero dentro del IC 95 %
        if k.startswith("E") or k == "alpha":
            assert u["valor"] == pytest.approx(v, rel=tol), k
    assert res["arranques"] is None or res["arranques"]["coinciden"]


# Teoría
@pytest.mark.parametrize("test,model,d,classic,t", [
    ("relaxation", "FractionalMaxwellGel", {"E": 20.0, "tau": 10.0}, lambda t, E, tau: E * np.exp(-t / tau), np.logspace(0, np.log10(20), 20)),
    ("creep", "FractionalMaxwellGel", {"E": 20.0, "tau": 10.0}, lambda t, E, tau: (1 + t / tau) / E, np.logspace(0, np.log10(20), 20)),
    ("creep", "FractionalKelvinVoigtS", {"E": 20.0, "tau": 10.0}, lambda t, E, tau: (1 - np.exp(-t / tau)) / E, np.logspace(0, np.log10(20), 20)),
    # en relajación del KVF solo para t >> tau
    ("relaxation", "FractionalKelvinVoigtS", {"E": 20.0, "tau": 10.0}, lambda t, E, tau: E + 0 * t, np.logspace(2, 3, 20)),
])
def test_limite_alpha_1_da_los_modelos_clasicos(test, model, d, classic, t):
    # se compara en 0.1-2 tau, con el error referido a 1/E o E
    y = _eval(test, model, {**d, "alpha": 0.999}, t)
    ref = classic(t, d["E"], d["tau"])
    scale = 1 / d["E"] if test == "creep" else d["E"]
    assert np.max(np.abs(y - ref)) / scale < 1e-3


@pytest.mark.parametrize("test,ts", [("relaxation", "tau_sigma"), ("creep", "tau_epsilon")])
def test_limite_alpha_1_del_zfm_es_el_zener(test, ts):
    t = np.logspace(-1, 2, 30)
    d = {"E_r": 10.0, "E_u": 30.0, ts: 5.0}
    y = _eval(test, "FractionalZenerSolidS", {**d, "alpha": 0.999}, t)
    ref = _eval(test, "Zener", d, t)
    assert np.max(np.abs(y / ref - 1)) < 1e-3


def _interconversion(model, p, t):
    g = lambda u: engine.model_eval("relaxation", model, p, [u])[0]
    j = lambda u: engine.model_eval("creep", model, p, [u])[0]
    total, a = 0.0, 0.0
    for b in [t * q for q in (1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99)] + [t]:
        total += quad(lambda u: g(u) * j(t - u), a, b, limit=200)[0]
        a = b
    return total / t


@pytest.mark.parametrize("model,d", [
    ("FractionalKelvinVoigtS", {"E": 20.0, "tau": 2.0, "alpha": 0.35}),
    ("FractionalZenerSolidS", {"E_r": 10.0, "E_u": 30.0, "tau_sigma": 2.0, "alpha": 0.35}),
])
def test_interconversion_en_parametros_del_doctor(model, d):
    # integral_0^t E(t - s) D(s) ds = t: creep y relajación describen el mismo material
    p = md.to_native("relaxation", model, d)
    for t in (1.0, 4.0):
        assert _interconversion(model, p, t) == pytest.approx(1.0, abs=2e-3)


def test_valores_en_t_0_y_t_infinito():
    a = 0.35
    fmm = {"E": 20.0, "tau": 10.0, "alpha": a}
    assert _eval("relaxation", "FractionalMaxwellGel", fmm, [1e-20 * 10])[0] == pytest.approx(20.0, rel=1e-6)   # E(0+) = E
    assert _eval("creep", "FractionalMaxwellGel", fmm, [1e-20 * 10])[0] == pytest.approx(1 / 20.0, rel=1e-6)    # D(0) = 1/E
    kvf = {"E": 20.0, "tau": 10.0, "alpha": a}
    assert _eval("creep", "FractionalKelvinVoigtS", kvf, [1e20 * 10])[0] == pytest.approx(1 / 20.0, rel=1e-6)   # D(inf) = 1/E
    zc = {"E_r": 10.0, "E_u": 30.0, "tau_epsilon": 20.0, "alpha": a}
    assert _eval("creep", "FractionalZenerSolidS", zc, [1e-20 * 20])[0] == pytest.approx(1 / 30.0, rel=1e-6)    # D(0) = 1/E_u
    assert _eval("creep", "FractionalZenerSolidS", zc, [1e20 * 20])[0] == pytest.approx(1 / 10.0, rel=1e-6)     # D(inf) = 1/E_r
    zr = {"E_r": 10.0, "E_u": 30.0, "tau_sigma": 5.0, "alpha": a}
    assert _eval("relaxation", "FractionalZenerSolidS", zr, [1e-20 * 5])[0] == pytest.approx(30.0, rel=1e-6)    # E(0+) = E_u
    assert _eval("relaxation", "FractionalZenerSolidS", zr, [1e20 * 5])[0] == pytest.approx(10.0, rel=1e-6)     # E(inf) = E_r


def test_conversion_doctor_pyrheo_ida_y_vuelta():
    for test, model, d in CASES:
        prim, _ = md.to_doctor(test, model, md.to_native(test, model, d))
        for k, v in d.items():
            assert prim[k] == pytest.approx(v, rel=1e-12)


def test_tiempos_del_zfm_cumplen_tau_e_alpha_igual_tau_s_alpha_eu_sobre_er():
    d = {"E_r": 10.0, "E_u": 30.0, "tau_sigma": 5.0, "alpha": 0.4}
    _, der = md.to_doctor("relaxation", "FractionalZenerSolidS", md.to_native("relaxation", "FractionalZenerSolidS", d))
    assert der["tau_epsilon"] ** 0.4 == pytest.approx(5.0 ** 0.4 * 30 / 10, rel=1e-12)


# Estadísticos
def test_estadisticos_del_doctor_coinciden_con_el_calculo_directo():
    rng = np.random.default_rng(11)
    d = {"E": 20.0, "tau": 10.0, "alpha": 0.35}
    t = np.linspace(0.5, 600, 3000)    # muchas observaciones: se promedian en intervalos
    y = _eval("relaxation", "FractionalMaxwellGel", d, t) * (1 + 0.005 * rng.standard_normal(len(t)))
    out = _fit_api("relaxation", t, y, ["FractionalMaxwellGel", "SpringPot", "FractionalKelvinVoigtS"])
    res = next(r for r in out["results"] if r["model"] == "FractionalMaxwellGel")
    yh = engine.model_eval("relaxation", "FractionalMaxwellGel", res["params"], t)
    T_ = res["todos"]
    assert T_["n"] == len(t)
    assert T_["aad_pct"] == pytest.approx(np.mean(np.abs((y - yh) / y)) * 100, rel=1e-4)
    assert T_["mse"] == pytest.approx(np.mean((y - yh) ** 2), rel=1e-4)
    assert T_["r2"] == pytest.approx(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2), rel=1e-6)
    ok = [r for r in out["results"] if "error" not in r]
    assert sum(r["peso_akaike"] for r in ok) == pytest.approx(1.0)
    n, k = out["n"], res["k"]
    assert res["aicc"] == pytest.approx(n * np.log(res["rss_w"] / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))
    assert res["s_res_pct"] == pytest.approx(np.sqrt(res["rss_w"] / (n - k)) * 100)
    assert abs(res["autocorrelacion_lag1"]) < 0.5          # residuos de ruido, sin tendencia
    sp = next(r for r in out["results"] if r["model"] == "SpringPot")
    assert sp["autocorrelacion_lag1"] > 0.8                # el modelo equivocado deja tendencia
    assert res["peso_akaike"] > 0.9


def test_matriz_de_correlacion_es_valida():
    rng = np.random.default_rng(3)
    d = {"E": 20.0, "tau": 10.0, "alpha": 0.35}
    y = _eval("creep", "FractionalMaxwellGel", d, T) * (1 + 0.01 * rng.standard_normal(len(T)))
    res = _fit_api("creep", T, y, ["FractionalMaxwellGel"])["results"][0]
    C = res["doctor"]["correlacion"]
    M = np.array(C["matriz"], float)
    assert C["parametros"] == ["E", "tau", "alpha"]
    assert np.allclose(np.diag(M), 1) and np.allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(M) > -1e-9)          # semidefinida positiva


def test_kvf_degenerado_se_reporta_como_springpot():
    # datos de Springpot ajustados con KVF: el resorte no aporta, E y tau no deben marcarse determinados
    y = engine.model_eval("creep", "SpringPot", {"V": 18.0, "alpha": 0.07}, T)
    res = _fit_api("creep", T, y, ["FractionalKelvinVoigtS"])["results"][0]
    P = res["doctor"]["parametros"]
    assert not P["E"]["determinado"] and not P["tau"]["determinado"]
    assert P["alpha"]["valor"] == pytest.approx(0.07, rel=1e-2)
    assert any("se reduce a un Springpot" in a for a in res["avisos"])


# Cobertura de los intervalos
def test_cobertura_del_ic95():
    """El IC 95 % contiene el valor verdadero en ~95 % de las réplicas."""
    reps = 200 if os.environ.get("SPRINGPLOT_SLOW") else 40
    d = {"E": 20.0, "tau": 10.0, "alpha": 0.35}
    t = np.logspace(-1, 3, 60)
    y0 = _eval("relaxation", "FractionalMaxwellGel", d, t)
    rng = np.random.default_rng(2026)
    hits = {k: 0 for k in d}
    for _ in range(reps):
        y = y0 * (1 + 0.01 * rng.standard_normal(len(t)))
        p, _, _ = varpro.fit("relaxation", "FractionalMaxwellGel", t, y)
        P = engine._doctor("relaxation", "FractionalMaxwellGel", p, t, y, np.ones_like(y))["parametros"]
        for k, v in d.items():
            hits[k] += P[k]["inf"] <= v <= P[k]["sup"]
    lo = 0.90 if reps >= 200 else 0.85
    for k, h in hits.items():
        assert lo <= h / reps <= 1.0, (k, h / reps)
