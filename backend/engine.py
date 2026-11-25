"""Motor de ajuste.

Dos rutas:
  1) pyRheo real (si se puede importar). Ver `_fit_pyrheo` y la sección VERIFICAR.
  2) Respaldo con SciPy (modelos clásicos simples) para probar la página sin pyRheo.
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import least_squares, minimize

import modelos_doctor
import varpro
from scipy.special import gamma as _gamma

IMPORT_ERROR=""
NUM_GUESSES = 16  # pyRheo usa 64 por defecto; 16 mantiene el tiempo de respuesta razonable en Render

try:  # --- VERIFICAR contra el repo: github.com/mirandi1/pyRheo ---
    from pyRheo.creep_model import CreepModel
    from pyRheo.relaxation_model import RelaxationModel
    from pyRheo.oscillation_model import SAOSModel
    from pyRheo.rotation_model import SteadyShearModel
    PYRHEO = True
except Exception as e:  # noqa
    PYRHEO = False
    IMPORT_ERROR= repr(e)


# Modelos de pyRheo (Tabla 2 del paper) por tipo de ensayo
PYRHEO_MODELS = {
    # Familia principal de un solo orden (notación del doctor): Springpot, FMM = FractionalMaxwellGel,
    # KVF = FractionalKelvinVoigtS, ZFM = FractionalZenerSolidS. Las de dos springpots quedan como prueba de robustez.
    "creep": ["auto", "SpringPot", "FractionalMaxwellGel", "FractionalKelvinVoigtS", "FractionalZenerSolidS",
              "Maxwell", "Zener", "FractionalMaxwellLiquid", "FractionalMaxwell", "FractionalKelvinVoigt", "FractionalZener"],
    "relaxation": ["auto", "SpringPot", "FractionalMaxwellGel", "FractionalKelvinVoigtS", "FractionalZenerSolidS",
                   "Maxwell", "Zener", "FractionalMaxwellLiquid", "FractionalMaxwell", "FractionalKelvinVoigt"],
    "saos": ["auto", "Maxwell", "SpringPot", "FractionalMaxwellGel", "FractionalMaxwellLiquid",
             "FractionalMaxwell", "FractionalKelvinVoigt", "FractionalZener"],
    "steady": ["HerschelBulkley", "Bingham", "PowerLaw", "CarreauYasuda", "Cross", "Casson"],
}

# ---------- Respaldo SciPy ----------
def _mx_relax(t, G, tau): return G * np.exp(-t / tau)
def _sp_relax(t, V, a): return V * t ** (-a) / _gamma(1 - a)
def _mx_creep(t, G, eta): return 1 / G + t / eta
def _sp_creep(t, V, a): return t ** a / (V * _gamma(1 + a))
def _mx_saos(w, G, tau):
    x = (w * tau) ** 2
    return np.concatenate([G * x / (1 + x), G * np.sqrt(x) / (1 + x)])
def _sp_saos(w, V, a):
    m = V * w ** a
    return np.concatenate([m * np.cos(np.pi * a / 2), m * np.sin(np.pi * a / 2)])
def _power(g, K, n): return K * g ** (n - 1)
def _cross(g, e0, einf, K, m): return einf + (e0 - einf) / (1 + (K * g) ** m)

# nombre -> (función, nombres de parámetros, límites inferiores, superiores)
FALLBACK = {
    "relaxation": {"Maxwell": (_mx_relax, ["G", "tau"], [1e-6, 1e-6], [1e12, 1e6]),
                   "SpringPot": (_sp_relax, ["V", "alpha"], [1e-6, 0.0], [1e12, 0.99])},
    "creep": {"Maxwell": (_mx_creep, ["G", "eta"], [1e-6, 1e-6], [1e12, 1e14]),
              "SpringPot": (_sp_creep, ["V", "alpha"], [1e-6, 0.0], [1e12, 0.99])},
    "saos": {"Maxwell": (_mx_saos, ["G", "tau"], [1e-6, 1e-6], [1e12, 1e6]),
             "SpringPot": (_sp_saos, ["V", "alpha"], [1e-6, 0.0], [1e12, 1.0])},
    "steady": {"PowerLaw": (_power, ["K", "n"], [1e-9, 0.0], [1e9, 2.0]),
               "Cross": (_cross, ["eta0", "eta_inf", "K", "m"], [1e-6, 0.0, 1e-6, 0.0], [1e9, 1e6, 1e6, 3.0])},
}


def available_models(test):
    return PYRHEO_MODELS[test] if PYRHEO else list(FALLBACK[test])


def _fit_scipy(test, model, x, y, restarts=10):
    if model not in FALLBACK[test]:
        raise ValueError(f"Modelo '{model}' no disponible en el modo de respaldo.")
    f, names, lo, hi = FALLBACK[test][model]
    lo, hi = np.array(lo, float), np.array(hi, float)
    resid = lambda p: (y - f(x, *p)) / y  # residual ponderado, igual que RSS_w del paper
    rng = np.random.default_rng(0)
    best = None
    for _ in range(restarts):
        # arranque aleatorio: log-uniforme si el rango es grande, uniforme si no
        p0 = np.where(hi / np.maximum(lo, 1e-12) > 1e3,
                      np.exp(rng.uniform(np.log(np.maximum(lo, 1e-12)), np.log(hi))),
                      rng.uniform(lo, hi))
        try:
            r = least_squares(resid, p0, bounds=(lo, hi))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("El ajuste no convergió.")
    return dict(zip(names, best.x)), (lambda xx: f(xx, *best.x))


ML_FIT = "Pade63"     # aproximación rápida para optimizar (la Padé 3/2 se degrada con alpha > 0.3)
ML_EXACT = "Garrappa"  # método de Garrappa: error < 1e-6 % frente a la serie de alta precisión
ML_TOL = 0.0005        # si la Padé difiere más de 0.05 % del método exacto, se repule el ajuste con el exacto
_EVAL_LOG = []         # evaluaciones en que el método exacto falló y se usó la Padé (se reporta por modelo)


def _module(test):
    import importlib
    return importlib.import_module({"saos": "pyRheo.oscillation_model", "relaxation": "pyRheo.relaxation_model",
                                    "creep": "pyRheo.creep_model", "steady": "pyRheo.rotation_model"}[test])


def _fzss_creep(p, t, ml=ML_EXACT):
    """Creep del Zener fraccionario sólido (resorte G_p en paralelo con Maxwell fraccionario gel G_s, V, alpha).
    pyRheo 1.0.3 usa tau = (V/G_s)^(1/alpha), que es el tiempo de RELAJACIÓN; en creep el tiempo de retardo es
    tau_r = tau * ((G_p + G_s)/G_p)^(1/alpha) (con alpha = 1 se recupera el Zener clásico). Derivado por
    transformada de Laplace de la relajación y verificado con la identidad integral_0^t G(t-s) J(s) ds = t."""
    from pyRheo.mittag_leffler.ml_garrappa import E_alpha_beta
    from pyRheo.mittag_leffler.ml_pade63 import R_alpha_beta_6_3
    Gp, Gs, V, a = (float(p[k]) for k in ("G_p", "G_s", "V", "alpha"))
    tau_r = (V / Gs) ** (1 / a) * ((Gp + Gs) / Gp) ** (1 / a)
    t = np.asarray(t, float)
    z = -np.power(t / tau_r, a)
    with np.errstate(all="ignore"):
        if ml == ML_EXACT:
            try:
                e = np.real(np.asarray(E_alpha_beta(z, a, 1), complex))
                if not np.all(np.isfinite(e)):
                    raise ValueError
            except Exception:  # noqa
                _EVAL_LOG.append("FractionalZenerSolidS")
                e = np.real(np.asarray(R_alpha_beta_6_3(z, a, 1), complex))
        else:
            e = np.real(np.asarray(R_alpha_beta_6_3(z, a, 1), complex))
    return 1 / (Gp + Gs) + Gs / (Gp * (Gp + Gs)) * (1 - e)


def model_eval(test, model, params, x, ml=ML_EXACT):
    """Evalúa un modelo de pyRheo con parámetros dados. Para SAOS devuelve G' y G'' concatenados."""
    if test == "creep" and model == "FractionalZenerSolidS":
        return _fzss_creep(params, x, ml)
    if model == "Cross" and "a" not in params:
        params = {**params, "a": 1.0}  # pyRheo pide "a" aunque su ecuación de Cross no lo usa
    mod = _module(test)
    if model not in mod.MODEL_FUNCS:
        raise ValueError(f"Modelo '{model}' no disponible para esta respuesta.")
    fn, names = mod.MODEL_FUNCS[model], mod.MODEL_PARAMS[model]
    missing = [n for n in names if n not in params]
    if missing:
        raise ValueError("Faltan parametros: " + ", ".join(missing))
    args = [float(params[n]) for n in names]
    x = np.asarray(x, float)
    uses_ml = "mittag_leffler_type" in fn.__code__.co_varnames

    def run(kind):
        with np.errstate(all="ignore"):
            y = fn(*args, x, mittag_leffler_type=kind) if uses_ml else fn(*args, x)
        return np.real(np.asarray(y, complex)).astype(float)

    try:
        y = run(ml)
        if uses_ml and not np.all(np.isfinite(y)):
            raise ValueError("resultado no finito")
    except Exception:
        if not uses_ml or ml != ML_EXACT:
            raise
        # El algoritmo de Garrappa falla cuando el orden de la Mittag-Leffler es casi cero (p. ej. alpha - beta ~ 0
        # en Kelvin-Voigt fraccionario). En esa región la Padé 6/3 tiene error < 1e-8 %.
        _EVAL_LOG.append(model)
        y = run(ML_FIT)
    return y


def _polish(test, model, params, x, y, maxfev=250, w=None):
    """Reajuste local con la Mittag-Leffler exacta, partiendo de los parámetros obtenidos con la Padé."""
    names = list(params)
    frac = [n in ("alpha", "beta", "kappa") for n in names]
    eps = 1e-9

    def to_th(p):
        out = []
        for n, f in zip(names, frac):
            v = float(p[n])
            if f:
                v = min(max(v, eps), 1 - eps)
                out.append(np.log(v / (1 - v)))
            else:
                out.append(np.log(max(v, 1e-300)))
        return np.array(out)

    def from_th(th):
        return {n: (float(1 / (1 + np.exp(-v))) if f else float(np.exp(v))) for n, v, f in zip(names, th, frac)}

    ww = np.ones_like(y) if w is None else w

    def cost(th):
        r = (y - model_eval(test, model, from_th(th), x)) / y
        c = float(np.sum(ww * r ** 2))
        return c if np.isfinite(c) else 1e300

    th0 = to_th(params)
    c0 = cost(th0)
    r = minimize(cost, th0, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-7, "fatol": 1e-12})
    return (from_th(r.x), True) if r.fun < c0 else (dict(params), False)


def _fit_pyrheo(test, model, x, y, n_guesses=None, w=None):
    """VERIFICAR: nombres de clases, argumentos y atributos según el repo/documentación de pyRheo."""
    cls = {"creep": CreepModel, "relaxation": RelaxationModel,
           "saos": SAOSModel, "steady": SteadyShearModel}[test]
    # Verificado contra el codigo fuente de pyRheo: el argumento es `initial_guesses` (plural);
    # SAOS y SteadyShear NO aceptan `mittag_leffler_type`; el algoritmo por defecto
    # de SAOS/Steady es Nelder-Mead y el de creep/relajacion es Powell.
    kw = dict(model=model, initial_guesses="random", num_initial_guesses=n_guesses or NUM_GUESSES)
    if w is not None and not np.allclose(w, 1):
        kw["cost_function"] = ("RSS_custom", y / np.sqrt(w))  # sum(((y - yhat) / (y / sqrt(w)))^2) = sum(w ((y - yhat)/y)^2)
    if test in ("creep", "relaxation"):
        kw.update(minimization_algorithm="Powell", mittag_leffler_type=ML_FIT)
    m = cls(**kw)
    if test == "saos":
        m.fit(x, y[:len(x)], y[len(x):])
        pred = lambda xx: np.concatenate(m.predict(xx))
    else:
        m.fit(x, y)
        pred = lambda xx: np.asarray(m.predict(xx))
    if getattr(m, "params_", None) is None:
        raise RuntimeError("pyRheo no encontro una solucion para este modelo.")
    # get_parameters() agrega "Cost" y "Cost Metric" (texto): se descartan.
    params = {k: v for k, v in m.get_parameters().items() if k not in ("Cost", "Cost Metric")}
    return params, pred, m.model  # m.model queda con el modelo elegido si se uso "auto"


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def fit_settings(test):
    """Configuracion con la que se ajusta, para incluirla en el JSON descargable (transparencia y replicabilidad)."""
    if not PYRHEO:
        return {"motor": "SciPy least_squares (modo de respaldo, sin pyRheo)", "funcion_de_costo": "residual ponderado (y - f)/y",
                "arranques_aleatorios": 10, "semilla": 0}
    try:
        from importlib.metadata import version
        ver = version("pyRheo")
    except Exception:  # noqa
        ver = "desconocida"
    s = {"motor": "pyRheo", "version_pyrheo": ver, "repositorio": "https://github.com/mirandi1/pyRheo",
         "initial_guesses": "random", "num_initial_guesses": NUM_GUESSES, "cost_function": "RSS (ponderado por y)",
         "minimization_algorithm": "Powell" if test in ("creep", "relaxation") else "Nelder-Mead",
         "semillas": list(SEEDS),
         "reintentos": f"primer intento: semilla {ATTEMPTS[0][0]} de numpy.random con {ATTEMPTS[0][1]} valores iniciales; si R² log < "
                       f"{RETRY_R2_LOG} se repite con " + " y ".join(f"semilla {a} ({n} valores iniciales)" for a, n in ATTEMPTS[1:]) +
                       " y se conserva el intento de menor RSS_w. La semilla de cada modelo se reporta en sus resultados: con el "
                       "mismo archivo y configuración el ajuste es idéntico. El optimizador es local: un R² log bajo puede ser un "
                       "mínimo local y no una limitación del modelo; los residuos ayudan a distinguirlo.",
         "limite_por_modelo_s": MODEL_TIMEOUT, "limite_por_solicitud_s": TOTAL_BUDGET, "ajustes_en_paralelo": PARALLEL}
    if test in ("creep", "relaxation"):
        s["mittag_leffler"] = (f"la optimización usa la aproximación {ML_FIT}; curvas, métricas y residuos se calculan con el método "
                               f"exacto ({ML_EXACT}). Si ambos difieren más de {ML_TOL * 100:g} % en los datos, los parámetros se "
                               "repulen con el método exacto (Nelder-Mead, máx. 250 evaluaciones).")
    s["proyeccion_de_variables"] = ("además de la búsqueda de pyRheo, cada modelo se ajusta calculando exactos (mínimos cuadrados no "
                                    "negativos) los coeficientes lineales —módulos, viscosidades y cuasi-propiedades— y optimizando solo "
                                    "los órdenes fraccionarios y el tiempo característico; se conserva el resultado de menor pérdida. "
                                    "Evita los límites de pyRheo, que acotan viscosidades y cuasi-propiedades con un rango en unidades de módulo")
    s["correcciones_pyrheo"] = ["creep de FractionalZenerSolidS con el tiempo de retardo correcto (pyRheo 1.0.3 usa el de relajación)",
                                "las curvas se evalúan con la misma función con la que se reportan las métricas",
                                "límites dimensionales de viscosidades y cuasi-propiedades (ver proyeccion_de_variables)",
                                "Cross: el parámetro 'a' no interviene en la ecuación de pyRheo; no se ajusta ni se reporta"]
    s["notacion_doctor"] = ("Springpot, FMM, KVF y ZFM se reportan en la notación de tensión de Segura-Méndez, Puente-Córdova et al. "
                            "(2025): springpot sigma = E tau^alpha D^alpha epsilon. Ecuaciones de Mainardi y Spada (2011). Los IC 95 %, "
                            "errores estándar y la matriz de correlación se calculan directamente en (E, tau, alpha) o "
                            "(E_r, E_u, tau, alpha); el tiempo del ZFM que no es primario (tau_sigma en creep, tau_epsilon en "
                            "relajación) se obtiene por el método delta")
    s["estadisticos"] = {"R2": "1 - sum(y - yhat)^2 / sum(y - ybar)^2, sobre todas las observaciones",
                         "AAD_pct": "(1/N) sum |(y - yhat)/y| x 100, sobre todas las observaciones",
                         "MSE": "(1/N) sum (y - yhat)^2, sobre todas las observaciones (unidades de y al cuadrado)",
                         "s_res_pct": "sqrt(RSS_w/(n - k)) x 100, con n = intervalos ajustados",
                         "autocorrelacion_lag1": "correlación entre residuos relativos consecutivos (en orden de x)",
                         "AICc": "n ln(RSS_w/n) + 2k + 2k(k+1)/(n-k-1); pesos de Akaike w_i proporcionales a exp(-Delta_i/2)",
                         "BIC": "n ln(RSS_w/n) + k ln(n)",
                         "correlacion": f"rho_ij = C_ij/sqrt(C_ii C_jj); |rho| > {RHO_HIGH:g} se señala como alta",
                         "arranques": "parámetros a los que llegan los 3 mejores arranques de la proyección de variables; "
                                      "coinciden si su pérdida es la mínima o si caen dentro del IC 95 % del resultado"}
    return s


def predict_response(model, params, target, x):
    """Evalua un modelo ya ajustado en otro tipo de respuesta (Mittag-Leffler exacta).
    pyRheo usa los mismos nombres de modelo y de parametros en creep, relajacion y oscilacion; se verificó
    numéricamente que sus funciones son transformadas exactas entre sí (salvo el creep de Zener sólido S,
    que aquí se calcula con la expresión corregida).
    target: "saos" -> (G', G'') ; "relaxation" -> (G(t),) ; "creep" -> (J(t),)"""
    if not PYRHEO:
        raise ValueError("La prediccion necesita pyRheo instalado en el servidor.")
    y = model_eval(target, model, params, x)
    if target == "saos":
        half = len(y) // 2
        return y[:half], y[half:]
    return (y,)


def _normalize(model, params):
    """Maxwell y Kelvin-Voigt fraccionarios tienen dos springpots intercambiables (en serie o en paralelo): el modelo con
    (V, alpha) <-> (G, beta) es idéntico. Se reporta siempre con alpha >= beta, como en pyRheo y la literatura."""
    if model in ("FractionalMaxwell", "FractionalKelvinVoigt") and params.get("beta", 0) > params.get("alpha", 1):
        params = {**params, "V": params["G"], "G": params["V"], "alpha": params["beta"], "beta": params["alpha"]}
    return params


def _solid_tau(test, model, p):
    """Tiempo característico de Zener / Zener fraccionario sólido: de relajación tau = (V/G_s)^(1/alpha) (Zener: eta_s/G_s);
    en creep, el de retardo tau_r = tau ((G_p + G_s)/G_p)^(1/alpha), que es el que se observa en J(t)."""
    if model == "FractionalKelvinVoigtS":   # KVF: el mismo tau en creep (retardo) y relajación
        return (p["V"] / p["G_p"]) ** (1 / p["alpha"])
    a = 1.0 if model == "Zener" else p["alpha"]
    tau = p["eta_s"] / p["G_s"] if model == "Zener" else (p["V"] / p["G_s"]) ** (1 / a)
    return tau * ((p["G_p"] + p["G_s"]) / p["G_p"]) ** (1 / a) if test == "creep" else tau


# Comportamiento a tiempos largos que SUPONE cada modelo (su estructura, no los datos)
CHARACTER = {
    "Maxwell": "líquido", "FractionalMaxwellLiquid": "líquido",
    "Zener": "sólido", "FractionalZenerSolidS": "sólido", "FractionalKelvinVoigtS": "sólido",
    "SpringPot": "ley de potencia", "FractionalMaxwellGel": "ley de potencia", "FractionalMaxwell": "ley de potencia",
    "FractionalKelvinVoigt": "ley de potencia", "FractionalZener": "ley de potencia",
}
CHARACTER_TXT = {
    "líquido": "a tiempos largos el modelo FLUYE: la complianza crece linealmente con t (viscosidad finita) y el módulo tiende a cero",
    "sólido": "a tiempos largos el modelo llega a un EQUILIBRIO: módulo de equilibrio G_p y complianza acotada 1/G_p",
    "ley de potencia": "el modelo no supone equilibrio ni flujo: la respuesta sigue una ley de potencia y no afirma qué pasa "
                       "fuera de la ventana medida",
}
SOLID_REACHED = 0.9   # fracción de la respuesta de equilibrio que debe alcanzarse para decir que la meseta se observa


def _long_term(test, model, params, x, unc=None):
    """¿Los datos respaldan el comportamiento a tiempos largos que supone el modelo?
    T = duración del ensayo (creep, relajación) o 1/omega_min (SAOS).
    - líquido: el flujo se observa si su tiempo característico tau <= T (en creep se informa además qué fracción
      de J(T) aporta el amortiguador).
    - sólido: la meseta se observa si al final del ensayo la respuesta está al >= 90 % de la de equilibrio
      (creep: J(T) G_p; relajación: G_p / G(T); SAOS: G_p / G'(omega_min)) y G_p está determinado.
    - ley de potencia: no hace afirmación sobre tiempos largos.
    Criterio de interpretación, no normado por ASTM."""
    ch = CHARACTER.get(model)
    if ch is None or test == "steady":
        return None
    x = np.asarray(x, float)
    T = float(1 / x.min()) if test == "saos" else float(x.max())
    unc = unc or {}
    det = lambda *names: all(unc.get(n, {}).get("determinado", True) for n in names)
    out = {"caracter": ch, "descripcion": CHARACTER_TXT[ch], "ventana_s": T, "respaldado": None}
    try:
        with np.errstate(all="ignore"):
            if ch == "líquido":
                tau = (params["eta_s"] / params["G_s"] if model == "Maxwell"
                       else (params["eta_s"] / params["G"]) ** (1 / (1 - params["beta"])))
                out["tau_s"] = float(tau)
                frac = None
                if test == "creep":
                    frac = float((T / params["eta_s"]) / model_eval(test, model, params, np.array([T]))[0])
                    out["fraccion_flujo"] = frac
                ok = bool(tau <= T and det("eta_s"))
                out["respaldado"] = ok
                part = f"el amortiguador aporta {frac * 100:.1f} % de la complianza al final del ensayo y " if frac is not None else ""
                out["explicacion"] = (
                    (f"Flujo observado: {part}τ ≈ {tau:.3g} s ≤ duración ({T:.3g} s).") if ok else
                    (f"Flujo NO observado: {part}su tiempo característico τ ≈ {tau:.3g} s es {tau / T:.3g} veces la "
                     f"duración del ensayo ({T:.3g} s). Que el material fluya es una extrapolación del modelo, no un resultado "
                     "de los datos; un término así puede deberse también a deriva o deslizamiento en las mordazas."))
            elif ch == "sólido":
                if test == "creep":
                    frac = float(model_eval(test, model, params, np.array([T]))[0] * params["G_p"])
                    what = "la complianza medida al final es"
                elif test == "relaxation":
                    frac = float(params["G_p"] / model_eval(test, model, params, np.array([T]))[0])
                    what = "el módulo de equilibrio G_p es"
                else:
                    frac = float(params["G_p"] / model_eval(test, model, params, np.array([x.min()]))[0])
                    what = "el módulo de equilibrio G_p es"
                tau = _solid_tau(test, model, params)
                out["tau_s"], out["fraccion_equilibrio"] = float(tau), frac
                ok = bool(frac >= SOLID_REACHED and det("G_p"))
                out["respaldado"] = ok
                ref = "de la de equilibrio (1/G_p)" if test == "creep" else "del módulo al final del ensayo"
                out["explicacion"] = (
                    f"Meseta observada: {what} el {frac * 100:.0f} % {ref}." if ok else
                    f"Meseta NO observada: {what} solo el {frac * 100:.0f} % {ref} y el tiempo "
                    f"{'de retardo' if test == 'creep' else 'de relajación'} τ ≈ {tau:.3g} s "
                    f"({'≥' if tau >= T else '<'} duración {T:.3g} s). El equilibrio y el valor de G_p son extrapolaciones del modelo.")
            else:
                out["explicacion"] = ("Describe la ventana medida sin suponer meseta ni flujo; el comportamiento fuera de ella "
                                      "no está determinado por estos datos.")
    except Exception:  # noqa
        out["explicacion"] = "No se pudo evaluar."
    if model == "SpringPot" and "alpha" in params:
        a = float(params["alpha"])
        span = float(x.max() / x.min())
        out["interpretacion"] = (
            f"α = {a:.3g} (0 = resorte, 1 = amortiguador): "
            + ("casi elástico" if a < 0.2 else "casi viscoso" if a > 0.8 else "viscoelástico intermedio")
            + f". Factor de pérdida tan δ = tan(πα/2) = {np.tan(np.pi * a / 2):.3g}, igual en todas las frecuencias; "
            f"la respuesta varía un factor {span ** a:.3g} en las {np.log10(span):.1f} décadas medidas.")
    return out


def _diagnostics(test, model, params, x):
    """Señales de que los datos no determinan algún parámetro (el modelo se reduce a uno más simple)."""
    out = []
    for k, sym in (("alpha", "α"), ("beta", "β"), ("kappa", "κ")):
        v = params.get(k)
        if v is not None and v < 0.005:
            out.append(f"{sym} ≈ 0: ese springpot se comporta como un resorte.")
        elif v is not None and v > 0.995:
            out.append(f"{sym} ≈ 1: ese springpot se comporta como un amortiguador.")
    if model in ("FractionalZenerSolidS", "Zener") and {"G_p", "G_s"} <= set(params):
        if params["G_s"] < 1e-3 * params["G_p"]:
            out.append("E_u ≈ E_r: la rama viscoelástica casi no aporta y el modelo se reduce a un resorte.")
        elif params["G_p"] < 1e-3 * params["G_s"]:
            out.append("E_r ≈ 0: el resorte en paralelo casi no aporta y el modelo se reduce a FMM (Maxwell fraccionario).")
    if model == "FractionalKelvinVoigtS" and test in ("creep", "relaxation"):
        try:
            tau = (params["V"] / params["G_p"]) ** (1 / params["alpha"])
            if tau > 100 * float(np.max(x)):
                out.append("τ muy por encima de la ventana: el resorte en paralelo (E) casi no aporta y KVF se reduce a un "
                           "Springpot; E y τ no están determinados.")
            elif tau < float(np.min(x)) / 100:
                out.append("τ muy por debajo de la ventana: el springpot casi no aporta y KVF se reduce a un resorte.")
        except Exception:  # noqa
            pass
    if "alpha" in params and "beta" in params and abs(params["alpha"] - params["beta"]) < 0.005:
        out.append("α ≈ β: los dos springpots son casi iguales y el modelo se reduce a un Springpot; sus parámetros "
                   "individuales no están determinados por los datos.")
    tau_expr = {"Maxwell": lambda p: p["eta_s"] / p["G_s"], "FractionalMaxwellGel": lambda p: (p["V"] / p["G_s"]) ** (1 / p["alpha"]),
                "Zener": lambda p: _solid_tau(test, "Zener", p),
                "FractionalZenerSolidS": lambda p: _solid_tau(test, "FractionalZenerSolidS", p),
                "FractionalKelvinVoigtS": lambda p: _solid_tau(test, "FractionalKelvinVoigtS", p),
                "FractionalMaxwell": lambda p: (p["V"] / p["G"]) ** (1 / (p["alpha"] - p["beta"])),
                "FractionalMaxwellLiquid": lambda p: (p["eta_s"] / p["G"]) ** (1 / (1 - p["beta"]))}.get(model)
    if tau_expr is not None and test in ("creep", "relaxation", "saos"):
        try:
            with np.errstate(all="ignore"):
                tau = float(tau_expr(params))
            lo, hi = (1 / np.max(x), 1 / np.min(x)) if test == "saos" else (np.min(x), np.max(x))
            if np.isfinite(tau) and (tau > 100 * hi or tau < lo / 100):
                out.append(f"El tiempo característico τ ≈ {tau:.3g} s queda muy fuera de la ventana medida "
                           f"({lo:.3g}–{hi:.3g} s): los parámetros que dependen de él solo son cotas, no valores determinados.")
        except Exception:  # noqa
            pass
    return out


# Criterios de "parámetro determinado" (intervalo de confianza del 95 %, aproximación lineal)
DET_FACTOR = 2.0   # módulos, viscosidades, cuasi-propiedades: el intervalo no puede abarcar más de un factor 2 a cada lado
DET_ORDER = 0.1    # órdenes fraccionarios (alpha, beta, kappa, n): semiancho del intervalo <= 0.1
ORDER_NAMES = ("alpha", "beta", "kappa")
RHO_HIGH = 0.95   # |correlación| mayor: los dos parámetros se compensan y están mal determinados por separado


def _ci(names, is_ord, values, resid_fn, bounded=()):
    """Intervalos de confianza del 95 % por linealización (matriz de Fisher a partir del jacobiano del residuo
    relativo ponderado). Los parámetros con dimensiones se tratan en escala logarítmica (intervalo multiplicativo).
    Supone que el modelo describe bien la forma de los datos: si los residuos muestran tendencia, el intervalo
    real es más amplio. Un parámetro que el ajuste no puede fijar (p. ej. alpha ~ beta) da un intervalo enorme.
    Devuelve (intervalos por parámetro, covarianza en escala transformada o None, función th -> residuo, th0)."""
    from scipy.stats import t as student_t
    k = len(names)
    th0 = np.array([v if o else np.log(v) for v, o in zip(values, is_ord)], float)
    to_p = lambda th: {n: (v if o else float(np.exp(v))) for n, v, o in zip(names, th, is_ord)}
    r0 = resid_fn(to_p(th0))
    n = len(r0)
    J = np.empty((n, k))
    for j in range(k):
        h = 1e-5 if is_ord[j] else 1e-4
        tp, tm = th0.copy(), th0.copy()
        tp[j] += h
        tm[j] -= h
        J[:, j] = (resid_fn(to_p(tp)) - resid_fn(to_p(tm))) / (2 * h)
    dof = max(n - k, 1)
    s2 = float(r0 @ r0) / dof
    q = float(student_t.ppf(0.975, dof))
    cov, null_par = None, np.ones(k, bool)
    with np.errstate(all="ignore"):
        try:
            if not np.all(np.isfinite(J)):
                raise np.linalg.LinAlgError
            # Var(theta_j) = s2 * sum_i V_ji^2 / lambda_i. Las direcciones con lambda ~ 0 son combinaciones de parámetros
            # que los datos NO fijan (p. ej. alpha = beta, o un término que no aporta en la ventana medida): su varianza
            # es infinita. Una pseudo-inversa las pondría en cero y haría parecer "determinados" esos parámetros.
            lam, vec = np.linalg.eigh(J.T @ J)
            null = lam <= 1e-10 * max(lam.max(), 1e-300)
            cov = s2 * (vec[:, ~null] / lam[~null]) @ vec[:, ~null].T
            null_par = np.array([np.any(vec[j, null] ** 2 > 1e-6) for j in range(k)])
            var = np.where(null_par, np.inf, np.diag(cov))
            se = np.sqrt(var)
        except np.linalg.LinAlgError:
            se = np.full(k, np.inf)
    out = {}
    for j, name in enumerate(names):
        v = float(values[j])
        hw = q * se[j] if np.isfinite(se[j]) else np.inf
        if is_ord[j]:
            lo, hi = v - hw, v + hw
            at_bound = name in bounded and (v < 1e-3 or v > 1 - 1e-3)   # en el límite (0 o 1), el intervalo no es válido
            det = bool(np.isfinite(hw) and hw <= DET_ORDER and not at_bound)
            es = float(se[j]) if np.isfinite(se[j]) else None
        else:
            f = float(np.exp(hw)) if hw < 700 else np.inf
            lo, hi = (v / f, v * f) if np.isfinite(f) else (0.0, np.inf)
            det = bool(np.isfinite(f) and f <= DET_FACTOR)
            es = float(v * se[j]) if np.isfinite(se[j]) else None   # error estándar aproximado (lineal)
        out[name] = {"valor": v, "inf": float(lo) if np.isfinite(lo) else None, "sup": float(hi) if np.isfinite(hi) else None,
                     "error_estandar": es, "determinado": det}
    extra = {"cov": cov, "null": null_par, "q": q, "th0": th0, "to_p": to_p}
    return out, extra


def _correlation(names, extra):
    """Matriz de correlación de los parámetros (rho_ij = C_ij / sqrt(C_ii C_jj)); None en los no determinables."""
    cov, null = extra["cov"], extra["null"]
    if cov is None:
        return None
    d = np.sqrt(np.diag(cov))
    M = []
    for i in range(len(names)):
        row = []
        for j in range(len(names)):
            ok = not null[i] and not null[j] and d[i] > 0 and d[j] > 0
            row.append(float(np.clip(cov[i, j] / (d[i] * d[j]), -1, 1)) if ok else None)
        M.append(row)
    altas = [[names[i], names[j], M[i][j]] for i in range(len(names)) for j in range(i + 1, len(names))
             if M[i][j] is not None and abs(M[i][j]) > RHO_HIGH]
    return {"parametros": list(names), "matriz": M, "altas": altas}


def _delta(fun, extra, is_log=True):
    """Intervalo del 95 % de una cantidad derivada g(theta) por el método delta (gradiente numérico en la escala
    transformada); escala logarítmica para tiempos y módulos."""
    cov, null, th0, to_p, q = extra["cov"], extra["null"], extra["th0"], extra["to_p"], extra["q"]
    v = float(fun(to_p(th0)))
    if cov is None or not np.isfinite(v) or v <= 0:
        return {"valor": v, "inf": None, "sup": None, "error_estandar": None, "determinado": False}
    g = np.empty(len(th0))
    for j in range(len(th0)):
        tp, tm = th0.copy(), th0.copy()
        tp[j] += 1e-5
        tm[j] -= 1e-5
        g[j] = (np.log(fun(to_p(tp))) - np.log(fun(to_p(tm)))) / 2e-5
    if np.any(null & (np.abs(g) > 1e-8)):
        return {"valor": v, "inf": None, "sup": None, "error_estandar": None, "determinado": False}
    se = float(np.sqrt(max(g @ cov @ g, 0.0)))
    f = float(np.exp(q * se)) if q * se < 700 else np.inf
    ok = np.isfinite(f)
    return {"valor": v, "inf": v / f if ok else None, "sup": v * f if ok else None, "error_estandar": v * se,
            "determinado": bool(ok and f <= DET_FACTOR)}


def _uncertainty(test, model, params, x, y, w, extra_out=None):
    """IC 95 % de los parámetros de pyRheo (ver _ci)."""
    names = list(params)
    is_ord = [n in ORDER_NAMES or (model in ("PowerLaw", "HerschelBulkley", "Cross", "CarreauYasuda") and n in ("n", "a"))
              for n in names]
    sw = np.sqrt(w)
    out, extra = _ci(names, is_ord, [params[n] for n in names],
                     lambda p: sw * (y - model_eval(test, model, p, x, ml=ML_FIT)) / y, bounded=ORDER_NAMES)
    if extra_out is not None:
        extra_out.update(extra)
    return out


def _doctor(test, model, params, x, y, w):
    """Parámetros en la notación del doctor (E, tau, alpha; E_r, E_u, tau_sigma, tau_epsilon) con su IC 95 %,
    error estándar y matriz de correlación, calculados DIRECTAMENTE en esos parámetros (no traducidos después)."""
    conv = modelos_doctor.to_doctor(test, model, params)
    if conv is None:
        return None
    prim, _ = conv
    names = list(prim)
    is_ord = [modelos_doctor.KIND[n] == "order" for n in names]
    sw = np.sqrt(w)
    resid = lambda d: sw * (y - model_eval(test, model, modelos_doctor.to_native(test, model, d), x, ml=ML_FIT)) / y
    with np.errstate(all="ignore"):
        out, extra = _ci(names, is_ord, [prim[n] for n in names], resid, bounded=("alpha",))
        der = {k: _delta(lambda d, k=k: modelos_doctor.derived(test, model, d)[k], extra)
               for k in modelos_doctor.derived(test, model, prim)}
    return {"modelo": modelos_doctor.NAME.get(model, model), "parametros": out, "derivados": der,
            "correlacion": _correlation(names, extra)}


DEFAULT_TOL_PCT = 1.0  # incertidumbre de medición por defecto: ASTM E4 admite ±1 % de error en la fuerza indicada


def _rms_pct(r):
    t = r.get("todos") or {}
    if t.get("rms_pct") is not None:
        return float(t["rms_pct"])
    return float(np.sqrt(r["rss_w"] / max(r.get("n_fit", 1), 1)) * 100)


def recommend(results, n_points, tol_pct=DEFAULT_TOL_PCT):
    """Modelo recomendado (criterio de interpretación, NO normado por ASTM):
    1. Solo se consideran los modelos con TODOS sus parámetros determinados (si ninguno, todos).
    2. Equivalencia práctica: entre los que reproducen los datos con error RMS <= incertidumbre de medición (tol_pct),
       el de MENOS parámetros (empate: menor BIC). Una mejora menor que la incertidumbre del ensayo no puede
       atribuirse al material, y un término extra puede imponer un comportamiento (p. ej. flujo) que los datos no muestran.
    3. Si ninguno llega a esa incertidumbre: el de menor BIC = n ln(RSS_w / n) + k ln(n)."""
    ok = [r for r in results if "error" not in r and r.get("rss_w") is not None and r["rss_w"] > 0]
    if not ok:
        return None
    n = max(int(n_points), 2)
    for r in ok:
        r["n_fit"] = n
        r["k"] = len(r["params"])
        r["bic"] = float(n * np.log(r["rss_w"] / n) + r["k"] * np.log(n))
        k = r["k"]
        r["aicc"] = float(n * np.log(r["rss_w"] / n) + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf))
        r["todos_determinados"] = all(u["determinado"] for u in (r.get("incertidumbre") or {}).values())
        r["equivalente"] = _rms_pct(r) <= tol_pct
        lt = r.get("largo_plazo")
        if lt and lt.get("respaldado") and not r["equivalente"]:
            # la meseta o el flujo "observados" no valen si el modelo no reproduce los datos
            lt["respaldado"] = False
            lt["explicacion"] = lt["explicacion"].replace("observada:", "aparente:", 1).replace("observado:", "aparente:", 1)
            lt["explicacion"] += (f" Sin embargo, el modelo no reproduce los datos dentro de la incertidumbre de medición "
                                  f"(RMS {_rms_pct(r):.2g} % > {tol_pct:g} %): esa conclusión no es confiable.")
    det =[r for r in ok if r["todos_determinados"]]
    pool = det or ok
    equiv = [r for r in pool if r["equivalente"]]
    best_bic = min(pool, key=lambda r: r["bic"])
    if equiv:
        best = min(equiv, key=lambda r: (r["k"], r["bic"]))
        rule = "equivalencia_practica"
    else:
        best, rule = best_bic, "bic"
    overall = min(ok, key=lambda r: r["bic"])
    # pesos de Akaike (sobre todos los modelos ajustados con los mismos datos, ventana y pesos)
    amin = min(r["aicc"] for r in ok)
    if np.isfinite(amin):
        ex = {id(r): np.exp(-(r["aicc"] - amin) / 2) if np.isfinite(r["aicc"]) else 0.0 for r in ok}
        tot = sum(ex.values())
        for r in ok:
            r["delta_aicc"] = r["aicc"] - amin
            r["peso_akaike"] = float(ex[id(r)] / tot)
    for r in ok:
        r["delta_bic"] = r["bic"] - best["bic"]
        r["recomendado"] = r is best
    others = sorted((r for r in pool if r is not best), key=lambda r: r["bic"])
    lt = best.get("largo_plazo") or {}
    return {"modelo": best["model"], "k": best["k"], "con_parametros_determinados": bool(det),
            "regla": rule, "tolerancia_pct": float(tol_pct), "rms_pct": _rms_pct(best),
            "equivalentes": [r["model"] for r in sorted(equiv, key=lambda r: (r["k"], r["bic"]))],
            "menor_bic": best_bic["model"] if best_bic is not best else None,
            "delta_bic_siguiente": (others[0]["bic"] - best["bic"]) if others else None,
            "siguiente": others[0]["model"] if others else None,
            "mejor_ajuste_sin_restriccion": overall["model"] if overall is not best and overall is not best_bic else None,
            "caracter": lt.get("caracter"), "caracter_respaldado": lt.get("respaldado"),
            "menor_aicc": min(ok, key=lambda r: r["aicc"])["model"],
            "alcance": "mejor descripción de los datos DENTRO del rango medido; no es una clasificación del material",
            "criterio": (f"entre los modelos con todos sus parámetros determinados (IC 95 %: factor <= {DET_FACTOR:g} en módulos y "
                         f"viscosidades, semiancho <= {DET_ORDER:g} en órdenes), el de menos parámetros cuyo error RMS no supera la "
                         f"incertidumbre de medición indicada ({tol_pct:g} %): diferencias menores no pueden atribuirse al material. "
                         "Si ninguno la alcanza, el de menor BIC. Criterio de interpretación, no normado por ASTM")}


def _lag1(curves):
    """Autocorrelación de lag 1 de los residuos (en orden de x; en SAOS, promedio de G' y G'').
    Cerca de 0: ruido; cerca de 1: tendencia que el modelo no captura."""
    vals = []
    for r in curves:
        r = r[np.isfinite(r)]
        if len(r) > 3:
            d = r - r.mean()
            den = float(d @ d)
            if den > 0:
                vals.append(float(d[:-1] @ d[1:]) / den)
    return float(np.mean(vals)) if vals else None


def _starts_report(test, model, starts, unc):
    """¿Los 3 arranques de la proyección de variables llegan al mismo mínimo? Compara sus parámetros con el IC 95 %."""
    if not starts:
        return None
    best = min(starts, key=lambda s: s["perdida"])
    same = []
    for s in starts:
        ok = s["perdida"] <= best["perdida"] * (1 + 1e-6) + 1e-14
        if not ok and unc:
            ok = True
            for n, v in s["parametros"].items():
                u = unc.get(n) or {}
                if u.get("inf") is None or u.get("sup") is None or not (u["inf"] <= v <= u["sup"]):
                    ok = False
                    break
        same.append(bool(ok))
    return {"n": len(starts), "coinciden": all(same),
            "perdidas_relativas": [s["perdida"] / best["perdida"] if best["perdida"] > 0 else 1.0 for s in starts],
            "parametros": [s["parametros"] for s in starts]}


def _wloss(y, yhat, w):
    r = (y - yhat) / y
    v = float(np.sum(w * r ** 2))
    return v if np.isfinite(v) else float("inf")


def _fit_one(test, name, x, y, xd, n_guesses=None, w=None, x_all=None, y_all=None):
    """Ajusta un modelo y devuelve el dict de resultado (solo tipos simples, para poder cruzar procesos).
    w: pesos por punto (promedios por intervalo); x_all, y_all: todas las observaciones válidas, para métricas."""
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    ml, start = None, "aleatorio (pyRheo)"
    del _EVAL_LOG[:]
    starts = None
    only_vp = PYRHEO and test == "creep" and name == "FractionalZenerSolidS" and varpro.supported(test, name)
    if only_vp:
        # pyRheo 1.0.3 ajusta este creep con la fórmula equivocada (tiempo de relajación): solo proyección de variables
        params, loss, chosen = None, float("inf"), name
    elif PYRHEO:
        params, _, chosen = _fit_pyrheo(test, name, x, y, n_guesses, w)
        params = {k: float(v) for k, v in params.items() if not (chosen == "Cross" and k == "a")}
        loss = _wloss(y, model_eval(test, chosen, params, x), w)
    if PYRHEO:
        # proyección de variables: coeficientes lineales exactos (evita los límites de pyRheo sin dimensiones correctas)
        vp = varpro.fit(test, chosen, x, y, w)
        starts = None
        if vp:
            cand = vp[0]
            starts = vp[2] if len(vp) > 2 else None
            c_loss = _wloss(y, model_eval(test, chosen, cand, x), w)
            if c_loss < loss:
                params, loss, start = cand, c_loss, "proyección de variables (coeficientes lineales exactos)"
        if params is None:
            raise RuntimeError("El ajuste no convergió.")
        yhat = model_eval(test, chosen, params, x)
        if test in ("creep", "relaxation"):
            y_pade = model_eval(test, chosen, params, x, ml=ML_FIT)
            dev = float(np.nanmax(np.abs(y_pade / yhat - 1))) if np.all(np.isfinite(yhat)) else float("inf")
            ml = {"optimizacion": ML_FIT, "curvas_y_metricas": ML_EXACT, "desviacion_pade_pct": dev * 100, "repulido": False}
            if dev > ML_TOL:
                params, ml["repulido"] = _polish(test, chosen, params, x, y, w=w)
                yhat = model_eval(test, chosen, params, x)
        params = _normalize(chosen, params)
        pred = lambda xx: model_eval(test, chosen, params, xx)
    else:
        params, pred = _fit_scipy(test, name, x, y)
        chosen = name
        yhat = pred(x)
    yd = pred(xd)
    split = lambda v: [v[:len(v) // 2], v[len(v) // 2:]] if test == "saos" else [v]
    out = {
        "requested": name, "model": chosen,
        "params": {k: float(v) for k, v in params.items()},
        "r2": _r2(y, yhat), "r2_log": _r2(np.log10(y), np.log10(np.abs(yhat) + 1e-300)),
        "rss_w": _wloss(y, yhat, w),
        "curve_x": xd.tolist(), "curve_y": [c.tolist() for c in split(yd)],
        # residuo relativo en cada punto ajustado, en %: (dato - modelo) / dato
        "resid": [[float(v) if np.isfinite(v) else None for v in c] for c in split((y - yhat) / y * 100.0)],
        "mittag_leffler": ml,
        "inicio": start,
        "avisos": _diagnostics(test, chosen, params, x),
    }
    try:
        out["incertidumbre"] = _uncertainty(test, chosen, params, x, y, w) if PYRHEO else None
    except Exception:  # noqa
        out["incertidumbre"] = None
    try:
        out["doctor"] = _doctor(test, chosen, out["params"], x, y, w) if PYRHEO else None
    except Exception:  # noqa
        out["doctor"] = None
    out["largo_plazo"] = _long_term(test, chosen, out["params"], x, out["incertidumbre"]) if PYRHEO else None
    out["determinista"] = bool(only_vp)
    out["arranques"] = _starts_report(test, chosen, starts, out.get("incertidumbre"))
    # estadísticos del ajuste (sobre los promedios por intervalo, con los pesos del ajuste)
    k = len(out["params"])
    out["s_res_pct"] = float(np.sqrt(out["rss_w"] / max(len(y) - k, 1)) * 100)
    out["autocorrelacion_lag1"] = _lag1([np.asarray([v if v is not None else np.nan for v in c], float) for c in out["resid"]])
    # métricas sobre TODAS las observaciones válidas (no solo los promedios ajustados)
    if x_all is not None:
        x_all, y_all = np.asarray(x_all, float), np.asarray(y_all, float)
        # con muchas filas, la Padé 6/3 se usa solo si ya se comprobó que coincide con el método exacto en estos parámetros
        fast = ml is not None and len(x_all) > 5000 and ml["desviacion_pade_pct"] < ML_TOL * 100 and not ml["repulido"]
        ya = model_eval(test, chosen, params, x_all, ml=ML_FIT if fast else ML_EXACT) if PYRHEO else pred(x_all)
        fin = np.isfinite(ya) & np.isfinite(y_all)
        r = np.abs((y_all[fin] - ya[fin]) / y_all[fin]) * 100
        # métricas del doctor (Segura-Méndez et al. 2025, ec. 11-13) sobre todas las observaciones
        out["todos"] = {"n": int(len(y_all)), "rms_pct": float(np.sqrt(np.mean(r ** 2))) if len(r) else None,
                        "p95_pct": float(np.percentile(r, 95)) if len(r) else None,
                        "aad_pct": float(np.mean(r)) if len(r) else None,
                        "mse": float(np.mean((y_all[fin] - ya[fin]) ** 2)) if len(r) else None,
                        "r2": _r2(y_all[fin], ya[fin]) if len(r) > 1 else None,
                        "evaluador": (ML_FIT + " (validado contra Garrappa en estos parámetros)") if fast else
                                     (ML_EXACT if ml is not None else "forma cerrada")}
    if ml is not None:
        ml["evaluador_real"] = ML_EXACT if not _EVAL_LOG else f"{ML_FIT} en algunos puntos: {ML_EXACT} falló (orden de Mittag-Leffler ~ 0)"
    return out


def _worker(conn, test, name, x, y, xd, seed, n_guesses, w, x_all, y_all):
    import contextlib, io, warnings
    warnings.filterwarnings("ignore")
    try:
        # pyRheo genera sus valores iniciales con el generador global de NumPy: fijarlo hace el ajuste reproducible
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            conn.send(_fit_one(test, name, x, y, xd, n_guesses, w, x_all, y_all))
    except Exception as e:  # noqa
        conn.send({"requested": name, "error": str(e)})
    finally:
        conn.close()


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# Límites configurables con variables de entorno (en Railway: Variables del servicio).
# Los ajustes que nunca terminan (p. ej. Zener fraccionario de 6 parámetros en creep) se cortan aunque sobre CPU.
MODEL_TIMEOUT = _env_float("SPRINGPLOT_MODEL_TIMEOUT", 90)    # segundos por intento de cada modelo
TOTAL_BUDGET = _env_float("SPRINGPLOT_TOTAL_BUDGET", 240)     # segundos por solicitud completa
# Procesos de ajuste simultáneos en TODO el servidor (todas las solicitudes juntas). Ponerlo igual al número de
# vCPU del plan; no se usa os.cpu_count() porque en contenedores reporta los núcleos del equipo anfitrión.
PARALLEL = max(1, int(_env_float("SPRINGPLOT_PARALLEL", 2)))
_SLOTS = threading.BoundedSemaphore(PARALLEL)
_CTX = None
_CTX_LOCK = threading.Lock()


def _mp_context():
    """Contexto de multiprocessing seguro con hilos: 'forkserver' en Linux (hacer fork desde un proceso con varios
    hilos puede bloquearse) y 'spawn' en Windows. SPRINGPLOT_MP_START permite forzar otro ('spawn', 'fork')."""
    global _CTX
    import multiprocessing as mp
    with _CTX_LOCK:
        if _CTX is None:
            method = os.environ.get("SPRINGPLOT_MP_START") or ("forkserver" if sys.platform.startswith("linux") else "spawn")
            ctx = mp.get_context(method)
            if method == "forkserver":
                try:
                    ctx.set_forkserver_preload(["engine"])  # pyRheo, SciPy y NumPy se cargan una sola vez
                except Exception:  # noqa
                    pass
            _CTX = ctx
    return _CTX
# Intentos (semilla de numpy.random, número de valores iniciales aleatorios): el primero siempre; los demás solo
# si el ajuste sale pobre. El optimizador de pyRheo es local: con más valores iniciales es más probable hallar el
# mínimo global (en relajación, Zener sólido S pasó de R² log 0.82 con 16 a 0.995 con 64).
ATTEMPTS = ((0, NUM_GUESSES), (1, 64), (2, 64))
SEEDS = tuple(a[0] for a in ATTEMPTS)
RETRY_R2_LOG = 0.95  # si R2 log < este valor se reintenta con la siguiente semilla y se conserva el de menor RSS_w


def _attempt(test, name, x, y, xd, seed, n_guesses, limit, w=None, x_all=None, y_all=None):
    import time
    t0 = time.time()
    # espera un lugar libre (límite global de procesos); el tiempo de espera cuenta contra el presupuesto
    if not _SLOTS.acquire(timeout=max(limit, 0.1)):
        return {"requested": name, "error": "Servidor ocupado con otros ajustes; intenta de nuevo en un momento."}
    try:
        left = limit - (time.time() - t0)
        if left < 1:
            return {"requested": name, "error": "Omitido: se agotó el tiempo esperando un lugar libre en el servidor."}
        return _run_process(test, name, x, y, xd, seed, n_guesses, left, w, x_all, y_all)
    finally:
        _SLOTS.release()


def _run_process(test, name, x, y, xd, seed, n_guesses, limit, w, x_all, y_all):
    ctx = _mp_context()
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker, args=(child, test, name, x, y, xd, seed, n_guesses, w, x_all, y_all), daemon=True)
    p.start()
    child.close()
    if parent.poll(limit):
        try:
            res = parent.recv()
        except EOFError:
            res = {"requested": name, "error": "El ajuste terminó sin resultado (posible falta de memoria)."}
    else:
        res = {"requested": name, "error": f"Tardó más de {int(limit)} s y se canceló. Prueba con un modelo con menos parámetros."}
    if p.is_alive():
        p.terminate()
    p.join(2)
    return res


def run_fit(test, x, ys, models, w=None, x_all=None, ys_all=None):
    """x: array; ys: lista de arrays (1 para creep/relax/steady, 2 para SAOS: G', G'').
    Cada intento corre en su propio proceso con limite de tiempo y semilla fija (resultado reproducible).
    Si el ajuste sale pobre (R2 log < RETRY_R2_LOG) se repite con las semillas siguientes y se conserva el mejor."""
    import time
    y = np.concatenate(ys)
    w = None if w is None else (np.concatenate([w, w]) if test == "saos" and len(w) == len(x) else np.asarray(w, float))
    y_all = None if ys_all is None else np.concatenate(ys_all)
    xd = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    t_start = time.time()

    def one(name):
        best, tried = None, []
        for seed, n_guesses in ATTEMPTS:
            left = TOTAL_BUDGET - (time.time() - t_start)
            if left < 3:
                if best is None:
                    best = {"requested": name, "error": "Omitido: se agotó el tiempo total de la solicitud. Ajusta menos modelos a la vez."}
                break
            res = _attempt(test, name, x, y, xd, seed, n_guesses, min(MODEL_TIMEOUT, left), w, x_all, y_all)
            tried.append(seed)
            if "error" in res:
                if best is None:
                    best = res
                if "Tardó" in res["error"] or "ocupado" in res["error"] or not PYRHEO:
                    break          # si un intento no termina a tiempo, los demás tampoco lo harán
                continue
            if best is None or "error" in best or res["rss_w"] < best["rss_w"]:
                best = {**res, "semilla": seed, "valores_iniciales": n_guesses}
            if not PYRHEO or res["r2_log"] >= RETRY_R2_LOG or res.get("determinista"):
                break
        best["intentos"] = len(tried)
        best["semillas_probadas"] = tried
        return best

    with ThreadPoolExecutor(max_workers=max(1, min(PARALLEL, len(models)))) as pool:
        out = list(pool.map(one, models))
    out.sort(key=lambda r: r.get("rss_w", float("inf")))
    return out
