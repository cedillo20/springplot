"""Modelos hiperelásticos (Neo-Hookeano y Mooney-Rivlin) ajustados a la curva de tensión uniaxial.

NO NORMADO POR ASTM. ASTM D412 (tensión de hules vulcanizados y elastómeros termoplásticos) y D638 definen el ensayo,
pero ninguna norma ASTM define cómo ajustar constantes hiperelásticas. Se documenta aquí el procedimiento para que se
pueda reportar y reproducir.

Hipótesis (todas deben declararse en el reporte):
  - material incompresible e isótropo; tensión uniaxial homogénea, estiramiento lambda = 1 + epsilon (epsilon en
    fracción, medida desde el cero corregido del pie de curva de D638, Anexo A1);
  - respuesta elástica: se ajusta solo la rama de carga hasta el esfuerzo máximo (o hasta la cedencia si ocurre antes).
    El TPU es viscoelástico: las constantes valen para la velocidad de ensayo usada;
  - esfuerzo nominal P = F/A0 (el que reporta D638).

Energía de deformación y esfuerzo nominal en tensión uniaxial (Treloar; Ogden, Non-linear Elastic Deformations):
  Neo-Hookeano:   W = C10 (I1 - 3)                     P = 2 C10 (lambda - lambda^-2)
  Mooney-Rivlin:  W = C10 (I1 - 3) + C01 (I2 - 3)      P = 2 (C10 + C01/lambda) (lambda - lambda^-2)
Módulo de corte inicial mu = 2 (C10 + C01) y módulo de Young inicial E0 = 3 mu (incompresible).

Los dos modelos son LINEALES en sus constantes, así que el ajuste es por mínimos cuadrados lineales (solución exacta,
sin valores iniciales ni optimizador) sobre el esfuerzo nominal. El IC 95 % sale de la covarianza s^2 (A^T A)^-1 con
t de Student (n - k grados de libertad).

Gráfica de Mooney: P / [2 (lambda - lambda^-2)] contra 1/lambda. Mooney-Rivlin es una recta (ordenada C10,
pendiente C01); Neo-Hookeano es una horizontal. Si los puntos se curvan, ninguno de los dos describe el material.
"""
import numpy as np
from scipy.stats import t as student_t

ALCANCE = ("No normado por ASTM: D412 y D638 definen el ensayo de tensión, pero no el ajuste de modelos hiperelásticos. "
           "Supone material incompresible, isótropo y elástico; se ajusta la rama de carga hasta el esfuerzo máximo (o la "
           "cedencia). En un material viscoelástico como el TPU las constantes dependen de la velocidad de ensayo.")

MODELOS = {
    "NeoHookeano": {"parametros": ["C10"], "ecuacion": "P = 2 C10 (λ − λ⁻²)"},
    "MooneyRivlin": {"parametros": ["C10", "C01"], "ecuacion": "P = 2 (C10 + C01/λ)(λ − λ⁻²)"},
}


def _design(lam, model):
    g = 2.0 * (lam - lam ** -2)
    return np.column_stack([g]) if model == "NeoHookeano" else np.column_stack([g, g / lam])


def stress(model, params, strain_pct):
    """Esfuerzo nominal [MPa] de un modelo con constantes params (MPa) a deformaciones en %."""
    lam = 1.0 + np.asarray(strain_pct, float) / 100.0
    c = [params["C10"]] + ([params["C01"]] if model == "MooneyRivlin" else [])
    return _design(lam, model) @ np.array(c, float)


def _fit_one(lam, P, model):
    A = _design(lam, model)
    c, *_ = np.linalg.lstsq(A, P, rcond=None)
    res = P - A @ c
    n, k = len(P), A.shape[1]
    dof = max(n - k, 1)
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    q = float(student_t.ppf(0.975, dof))
    names = MODELOS[model]["parametros"]
    params = {nm: float(v) for nm, v in zip(names, c)}
    ic = {nm: {"valor": float(v), "error_estandar": float(e), "inf": float(v - q * e), "sup": float(v + q * e)}
          for nm, v, e in zip(names, c, se)}
    mu = 2.0 * sum(params.values())
    # error estándar de mu = 2 (C10 + C01): suma de las varianzas más la covarianza
    se_mu = 2.0 * float(np.sqrt(np.sum(cov)))
    ss_tot = float(np.sum((P - P.mean()) ** 2))
    rel = np.abs(res[P > 0] / P[P > 0]) * 100
    out = {
        "parametros": params, "ic95": ic,
        "mu": mu, "mu_ic95": [mu - q * se_mu, mu + q * se_mu], "E0": 3.0 * mu,
        "r2": 1.0 - float(res @ res) / ss_tot if ss_tot > 0 else None,
        "rms_MPa": float(np.sqrt(np.mean(res ** 2))),
        "aad_pct": float(np.mean(rel)) if len(rel) else None,
        "n": int(n),
        "avisos": [],
    }
    if model == "MooneyRivlin":
        rho = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        out["correlacion_C10_C01"] = rho
        neg = [k for k in ("C10", "C01") if params[k] < 0]
        if neg:
            out["avisos"].append(f"{' y '.join(neg)} < 0: el modelo reproduce la curva en el rango ajustado, pero no cumple "
                                 "la condición suficiente de estabilidad de Drucker (C10 ≥ 0 y C01 ≥ 0); no lo extrapoles "
                                 "fuera de ese rango ni a otros estados de esfuerzo (compresión, cortante, biaxial).")
        if abs(rho) > 0.95:
            out["avisos"].append(f"C10 y C01 están muy correlacionados (ρ = {rho:.3f}): los datos fijan bien su suma "
                                 "(μ) pero no cada constante por separado. Interpreta μ, no C10 y C01 individualmente.")
    if mu <= 0:
        out["avisos"].append("μ ≤ 0: el ajuste no tiene sentido físico.")
    return out


BARRIDO_PCT = (2, 5, 10, 20, 50, 100, 200, 400)


def fit(strain_pct, stress_MPa, end_index, modulo_D638=None, max_pct=None):
    """Ajusta Neo-Hookeano y Mooney-Rivlin a la rama de carga.
    strain_pct: deformación corregida (desde el cero del pie de curva), en %; end_index: último punto de la rama de carga;
    max_pct: deformación máxima a ajustar (None = toda la rama de carga)."""
    e_all = np.asarray(strain_pct, float)[: end_index + 1]
    s_all = np.asarray(stress_MPa, float)[: end_index + 1]
    ok = np.isfinite(e_all) & np.isfinite(s_all) & (e_all > 0) & (s_all > 0)
    e_all, s_all = e_all[ok], s_all[ok]
    sel = e_all <= max_pct if max_pct else np.ones(len(e_all), bool)
    e, s = e_all[sel], s_all[sel]
    if len(e) < 10:
        return {"error": "Menos de 10 puntos en la rama de carga: no se ajustan modelos hiperelásticos."}
    lam = 1.0 + e / 100.0
    out = {"rango_deformacion_pct": [float(e.min()), float(e.max())], "lambda_max": float(lam.max()),
           "n_puntos": int(len(e)), "alcance": ALCANCE, "modelos": {}, "limite_usuario_pct": max_pct,
           "fin_rama_carga_pct": float(e_all.max())}
    for m in MODELOS:
        r = _fit_one(lam, s, m)
        r["ecuacion"] = MODELOS[m]["ecuacion"]
        if modulo_D638:
            # comparación con el módulo de D638 (pendiente inicial): ambos deben coincidir en deformaciones pequeñas
            r["E0_vs_modulo_D638_pct"] = float((r["E0"] / modulo_D638 - 1) * 100)
        out["modelos"][m] = r
    # curvas para graficar (submuestreadas) y gráfica de Mooney
    idx = np.unique(np.linspace(0, len(e) - 1, min(len(e), 200)).astype(int))
    out["curvas"] = {"deformacion": e[idx].tolist(),
                     **{m: stress(m, out["modelos"][m]["parametros"], e[idx]).tolist() for m in MODELOS}}
    g = 2.0 * (lam - lam ** -2)
    keep = s > 0.05 * s.max()     # cerca de cero la división amplifica el ruido
    out["mooney"] = {"inv_lambda": (1 / lam[keep])[::max(1, keep.sum() // 200)].tolist(),
                     "esfuerzo_reducido": (s[keep] / g[keep])[::max(1, keep.sum() // 200)].tolist()}
    # Dependencia del rango: si las constantes cambian mucho al cambiar la deformación máxima ajustada, el modelo no
    # describe la forma de la curva y sus constantes solo valen para ese rango
    barrido = []
    for lim in [x for x in BARRIDO_PCT if x < e_all.max()] + [float(e_all.max())]:
        q = e_all <= lim
        if q.sum() < 10:
            continue
        lq = 1.0 + e_all[q] / 100.0
        row = {"hasta_pct": float(lim), "n": int(q.sum())}
        for m in MODELOS:
            rr = _fit_one(lq, s_all[q], m)
            row[m] = {"mu": rr["mu"], "aad_pct": rr["aad_pct"], **rr["parametros"]}
        barrido.append(row)
    out["barrido_rango"] = barrido
    mus = {m: [b[m]["mu"] for b in barrido] for m in MODELOS}
    out["variacion_mu_pct"] = {m: float((max(v) / min(v) - 1) * 100) if len(v) > 1 and min(v) > 0 else None for m, v in mus.items()}
    for m, v in out["variacion_mu_pct"].items():
        if v is not None and v > 20:
            out["modelos"][m]["avisos"].append(
                f"μ cambia {v:.0f} % según hasta qué deformación se ajuste (ver la tabla de dependencia del rango): el modelo no "
                "describe la forma de la curva y sus constantes solo valen para el rango ajustado. Repórtalas junto con ese rango.")
    if lam.max() < 1.2:
        out["aviso_rango"] = (f"La rama ajustada solo llega a λ = {lam.max():.3f} ({e.max():.1f} % de deformación). Los modelos "
                              "hiperelásticos se distinguen a deformaciones grandes; en este rango ambos se parecen a un "
                              "sólido lineal y C01 queda mal determinado.")
    return out


def summarize(results):
    """Promedio y desviación estándar entre probetas de las constantes (como 11.6 y 11.7 de D638)."""
    def stats(vals):
        v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
        if not len(v):
            return None
        return {"n": int(len(v)), "promedio": float(v.mean()), "desv_est": float(v.std(ddof=1)) if len(v) > 1 else None}
    H = [r.get("hiperelastico") or {} for r in results]
    g = lambda m, f: [f(h["modelos"][m]) for h in H if "modelos" in h]
    return {
        "NeoHookeano": {"C10": stats(g("NeoHookeano", lambda x: x["parametros"]["C10"])),
                        "mu": stats(g("NeoHookeano", lambda x: x["mu"]))},
        "MooneyRivlin": {"C10": stats(g("MooneyRivlin", lambda x: x["parametros"]["C10"])),
                         "C01": stats(g("MooneyRivlin", lambda x: x["parametros"]["C01"])),
                         "mu": stats(g("MooneyRivlin", lambda x: x["mu"]))},
    }
