"""Parámetros en la notación de tensión del grupo del doctor (Segura-Méndez, Puente-Córdova et al., 2025).

Springpot: sigma(t) = E tau^alpha D^alpha epsilon(t) (derivada de Caputo), tau = eta/E, 0 <= alpha <= 1.
Ecuaciones de creep y relajación de Mainardi y Spada (2011), reescritas en tensión:

  FMM (resorte E en serie con springpot E tau^alpha) = FractionalMaxwellGel de pyRheo
      E(t) = E E_alpha[-(t/tau)^alpha]                  D(t) = (1/E) [1 + (t/tau)^alpha / Gamma(1+alpha)]
  KVF (resorte E en paralelo con springpot E tau^alpha) = FractionalKelvinVoigtS de pyRheo
      E(t) = E [1 + (t/tau)^-alpha / Gamma(1-alpha)]    D(t) = (1/E) {1 - E_alpha[-(t/tau)^alpha]}
  ZFM (resorte E_r en paralelo con una rama FMM; E_u = módulo no relajado) = FractionalZenerSolidS de pyRheo
      E(t) = E_r + (E_u - E_r) E_alpha[-(t/tau_s)^alpha]
      D(t) = 1/E_u + (1/E_r - 1/E_u) {1 - E_alpha[-(t/tau_e)^alpha]},   tau_e^alpha = tau_s^alpha E_u/E_r

Equivalencias con pyRheo: G_s = E y V = E tau^alpha (FMM); G_p = E y V = E tau^alpha (KVF);
G_p = E_r, G_s = E_u - E_r y V = G_s tau_s^alpha (ZFM). Con alpha = 1 se recuperan Maxwell, Kelvin-Voigt y Zener.

Parámetro primario: el tiempo que se observa en el ensayo (en creep del ZFM, el de retardo tau_e; en relajación,
el de relajación tau_s); el otro se reporta como derivado, con su intervalo por el método delta.
"""
import numpy as np

# tipo de cada parámetro: "mod" (módulo, MPa), "time" (s), "order" (adimensional), "quasi" (MPa·s^alpha)
KIND = {"E": "mod", "E_r": "mod", "E_u": "mod", "tau": "time", "tau_sigma": "time", "tau_epsilon": "time",
        "alpha": "order", "V": "quasi"}

NAME = {"SpringPot": "Springpot (Scott Blair)", "FractionalMaxwellGel": "FMM", "FractionalKelvinVoigtS": "KVF",
        "FractionalZenerSolidS": "ZFM", "Maxwell": "Maxwell", "Zener": "Zener"}


def _fmm(p):
    return {"E": p["G_s"], "tau": (p["V"] / p["G_s"]) ** (1 / p["alpha"]), "alpha": p["alpha"]}


def _kvf(p):
    return {"E": p["G_p"], "tau": (p["V"] / p["G_p"]) ** (1 / p["alpha"]), "alpha": p["alpha"]}


def _zfm_common(p, a):
    Er, Eu = p["G_p"], p["G_p"] + p["G_s"]
    ts = (p["V"] / p["G_s"]) ** (1 / a) if "V" in p else p["eta_s"] / p["G_s"]
    return Er, Eu, ts, ts * (Eu / Er) ** (1 / a)


def to_doctor(test, model, p):
    """Parámetros de pyRheo -> (primarios, derivados) en notación del doctor; None si el modelo no está en la familia."""
    p = {k: float(v) for k, v in p.items()}
    with np.errstate(all="ignore"):
        if model == "SpringPot":
            return {"V": p["V"], "alpha": p["alpha"]}, {}
        if model == "FractionalMaxwellGel":
            return _fmm(p), {}
        if model == "FractionalKelvinVoigtS":
            return _kvf(p), {}
        if model == "Maxwell":
            return {"E": p["G_s"], "tau": p["eta_s"] / p["G_s"]}, {}
        if model in ("FractionalZenerSolidS", "Zener"):
            a = p.get("alpha", 1.0)
            Er, Eu, ts, te = _zfm_common(p, a)
            base = {"E_r": Er, "E_u": Eu}
            o = {"alpha": a} if model == "FractionalZenerSolidS" else {}
            if test == "creep":
                return {**base, "tau_epsilon": te, **o}, {"tau_sigma": ts}
            return {**base, "tau_sigma": ts, **o}, {"tau_epsilon": te}
    return None


def to_native(test, model, d):
    """Inverso de to_doctor (solo parámetros primarios)."""
    if model == "SpringPot":
        return {"V": d["V"], "alpha": d["alpha"]}
    if model == "FractionalMaxwellGel":
        return {"G_s": d["E"], "V": d["E"] * d["tau"] ** d["alpha"], "alpha": d["alpha"]}
    if model == "FractionalKelvinVoigtS":
        return {"G_p": d["E"], "V": d["E"] * d["tau"] ** d["alpha"], "alpha": d["alpha"]}
    if model == "Maxwell":
        return {"G_s": d["E"], "eta_s": d["E"] * d["tau"]}
    if model in ("FractionalZenerSolidS", "Zener"):
        a = d.get("alpha", 1.0)
        Er, Eu = d["E_r"], d["E_u"]
        ts = d["tau_sigma"] if "tau_sigma" in d else d["tau_epsilon"] * (Er / Eu) ** (1 / a)
        Gs = Eu - Er
        if model == "Zener":
            return {"G_p": Er, "G_s": Gs, "eta_s": Gs * ts}
        return {"G_p": Er, "G_s": Gs, "V": Gs * ts ** a, "alpha": a}
    raise ValueError(model)


def derived(test, model, d):
    """Cantidades derivadas a partir de los primarios (para su intervalo por el método delta)."""
    if model in ("FractionalZenerSolidS", "Zener"):
        a = d.get("alpha", 1.0)
        if test == "creep":
            return {"tau_sigma": d["tau_epsilon"] * (d["E_r"] / d["E_u"]) ** (1 / a)}
        return {"tau_epsilon": d["tau_sigma"] * (d["E_u"] / d["E_r"]) ** (1 / a)}
    return {}
