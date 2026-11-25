import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict

import tensile as tensile_d638
from engine import IMPORT_ERROR, PYRHEO, available_models, fit_settings, predict_response, recommend, run_fit

MAX_BYTES = 10 * 1024 * 1024
MAX_POINTS = 150  # puntos que se mandan al ajuste (muestreo logarítmico); los ensayos largos traen decenas de miles
FRONT = Path(__file__).resolve().parent.parent / "frontend"

# Columnas calculadas para el formato de los reómetros de FIME (esfuerzo en MPa, deformación en %)
J_NAME = "D(t) = Deformación/Tensión [1/MPa]"
G_NAME = "E(t) = Tensión/Deformación [MPa]"

app = FastAPI(title="SpringPlot")


def _norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


# ---------------------------------------------------------------- lectura
def read_raw(upload: UploadFile, raw: bytes, sheet: str = ""):
    """Lee el archivo SIN encabezado. Devuelve (DataFrame crudo, hojas, hoja usada)."""
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "El archivo pesa más de 10 MB.")
    name = (upload.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(io.BytesIO(raw))
            sheets = [str(s) for s in xl.sheet_names]
            used = sheet if sheet in sheets else sheets[0]
            return xl.parse(used, header=None), sheets, used
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=None)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=None, encoding="latin-1")
        return df, ["datos"], "datos"
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "No pude leer el archivo. Usa .xlsx, .xls o .csv.")


def generic_table(rdf: pd.DataFrame) -> pd.DataFrame:
    """Tabla normal: la primera fila son los encabezados."""
    seen, names = {}, []
    for i, v in enumerate(rdf.iloc[0]):
        n = str(v).strip() if pd.notna(v) and str(v).strip() else f"columna {i + 1}"
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    df = rdf.iloc[1:].reset_index(drop=True)
    df.columns = names
    return df.infer_objects()


def fime_meta(df: pd.DataFrame):
    """Nombres de las columnas de tiempo, esfuerzo y deformación de una muestra FIME."""
    def first(prefix):
        return next((c for c in df.columns if _norm(c).startswith(prefix)), None)
    return {"t": first("tiempo"), "s": first("tension"), "e": first("deformacion")}


def fime_samples(rdf: pd.DataFrame):
    """Formato de los reómetros de FIME:
       fila 1: M1, M2, ... (una etiqueta por muestra, cada una abre un bloque de columnas)
       fila 2: nombres (Tiempo, Fuerza, Desplazamiento, Tension, Deformacion)
       fila 3: unidades (seg, N, mm, N/mm2, %)  ·  fila 4 en adelante: datos.
       Devuelve {etiqueta: DataFrame} o None si el archivo no tiene ese formato."""
    if rdf.shape[0] < 4 or rdf.shape[1] < 2:
        return None
    r0 = [str(v).strip() if pd.notna(v) else "" for v in rdf.iloc[0]]
    starts = [i for i, v in enumerate(r0) if re.fullmatch(r"M\d+", v, re.I)]
    if not starts or not _norm(rdf.iloc[1, starts[0]]).startswith(("tiempo", "time")):
        return None
    out = {}
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else rdf.shape[1]
        names = []
        for j in range(s, e):
            n = str(rdf.iloc[1, j]).strip() if pd.notna(rdf.iloc[1, j]) else ""
            u = str(rdf.iloc[2, j]).strip() if pd.notna(rdf.iloc[2, j]) else ""
            names.append(f"{n} [{u}]" if n and u else (n or f"columna {j + 1}"))
        sub = rdf.iloc[3:, s:e].apply(pd.to_numeric, errors="coerce")
        sub.columns = names
        sub = sub.dropna(how="all").reset_index(drop=True)
        m = fime_meta(sub)
        if not (m["t"] and m["s"] and m["e"]):
            return None  # no trae tiempo, esfuerzo y deformación: se trata como tabla normal
        eps, sig = sub[m["e"]] / 100.0, sub[m["s"]]
        sub[J_NAME] = (eps / sig).where(sig > 0)
        sub[G_NAME] = (sig / eps).where(eps > 0)
        out[r0[s].upper()] = sub
    return out


def load_table(upload, raw, sheet="", sample=""):
    rdf, sheets, used = read_raw(upload, raw, sheet)
    fs = fime_samples(rdf)
    if fs:
        cur = sample if sample in fs else next(iter(fs))
        return {"df": fs[cur], "sheets": sheets, "sheet": used, "samples": list(fs), "sample": cur, "fime": True, "all": fs}
    if rdf.shape[0] < 2:
        raise HTTPException(400, "El archivo no tiene datos.")
    return {"df": generic_table(rdf), "sheets": sheets, "sheet": used, "samples": [], "sample": "", "fime": False, "all": None}


# ---------------------------------------------------------------- detección del ensayo
def classify_fime(df: pd.DataFrame) -> str:
    """creep: esfuerzo constante y la deformación crece · relajación: deformación constante y el
       esfuerzo cae · tensión: ninguno de los dos se mantiene (ensayo a velocidad constante hasta romper)."""
    m = fime_meta(df)
    d = df[[m["s"], m["e"]]].dropna()
    h = len(d) // 2
    if len(d) < 20:
        return ""
    cv = lambda v: float(np.std(v) / abs(np.mean(v))) if np.mean(v) else np.inf
    cvs, cve = cv(d[m["s"]].to_numpy()[h:]), cv(d[m["e"]].to_numpy()[h:])  # segunda mitad del ensayo
    if min(cvs, cve) > 0.03:
        return "tension"
    return "relaxation" if cve < cvs else "creep"


def guess_columns(columns):
    """Adivina ensayo y columnas por el nombre del encabezado (tablas que no son del reómetro FIME)."""
    low = [str(c).lower() for c in columns]

    def find(pattern):
        for c, l in zip(columns, low):
            if re.search(pattern, l):
                return str(c)
        return None

    gdot, eta = find(r"shear.?rate|γ|gamma|rate|veloc"), find(r"visc|η|eta")
    w, g1, g2 = find(r"^w|ω|omega|freq|frec"), find(r"g'|g′|storage|almac|^g1|g_p"), find(r"g''|g″|loss|p[eé]rd|^g2|g_pp")
    t = find(r"^t|time|tiempo")
    j, g = find(r"^j|compl"), find(r"^g|modul|módul")
    if gdot and eta:
        return {"test": "steady", "x": gdot, "y": eta}
    if w and g1 and g2:
        return {"test": "saos", "x": w, "y": g1, "y2": g2}
    if t and j:
        return {"test": "creep", "x": t, "y": j}
    if t and g:
        return {"test": "relaxation", "x": t, "y": g}
    return {}


def guess_fime(df):
    m, kind = fime_meta(df), classify_fime(df)
    if kind == "tension":
        return {"test": "tension", "x": m["e"], "y": m["s"]}
    if kind == "creep":
        return {"test": "creep", "x": m["t"], "y": J_NAME}
    if kind == "relaxation":
        return {"test": "relaxation", "x": m["t"], "y": G_NAME}
    return {}


# ---------------------------------------------------------------- preprocesado
def fime_segment(df, test):
    """Corrección por la rampa de carga (superposición de Boltzmann).
    El ensayo ideal aplica la carga de golpe; la máquina la aplica en una rampa de duración t_r. Para tiempos
    posteriores a la rampa, medir el tiempo desde la MITAD de la rampa reproduce la respuesta a una carga
    instantánea con error menor a 1 % (simulado con el modelo ajustado a los datos de FIME: con el origen al
    final de la rampa el sesgo llegaba a 11 % en relajación y 4.5 % en creep).
    La variable controlada define la rampa: deformación en relajación, esfuerzo en creep.
    Inicio de la rampa: primer punto >= 2 % del valor sostenido; fin: primer punto >= 98 %.
    Se conservan los datos desde el fin de la rampa. Devuelve (datos, info de la rampa)."""
    m = fime_meta(df)
    ctrl = m["e"] if test == "relaxation" else m["s"]
    d = df.dropna(subset=[m["t"], m["s"], m["e"]]).reset_index(drop=True)
    v = d[ctrl].to_numpy(float)
    t = d[m["t"]].to_numpy(float)
    hold = float(np.median(v[len(v) // 2:]))
    i_a = int(np.argmax(v >= 0.02 * hold))
    i_b = int(np.argmax(v >= 0.98 * hold))
    t0 = (t[i_a] + t[i_b]) / 2
    out = d.iloc[i_b:].copy()
    out[m["t"]] = out[m["t"]] - t0
    return out, {"inicio_s": float(t[i_a]), "fin_s": float(t[i_b]), "duracion_s": float(t[i_b] - t[i_a]), "origen_s": float(t0)}


def thin(data, xcol, n=MAX_POINTS):
    """Reduce a ~n puntos espaciados en escala logarítmica (data ya viene ordenada por xcol, x > 0)."""
    if len(data) <= n:
        return data
    x = data[xcol].to_numpy(float)
    targets = np.logspace(np.log10(x.min()), np.log10(x.max()), n)
    idx = np.unique(np.clip(np.searchsorted(x, targets), 0, len(x) - 1))
    return data.iloc[idx]


def log_bin(data, xcol, ycols, n=MAX_POINTS):
    """Promedia TODAS las observaciones en n intervalos de igual ancho en log10(x).
    Devuelve (tabla de promedios, número de observaciones por intervalo). x se promedia geométricamente.
    Con anchos de intervalo pequeños (~60 por década) el sesgo de promediar una curva suave es despreciable."""
    if len(data) <= n:
        return data.reset_index(drop=True), np.ones(len(data))
    x = data[xcol].to_numpy(float)
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n + 1)
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)
    g = pd.DataFrame({"b": b, "lx": np.log(x), **{c: data[c].to_numpy(float) for c in ycols}}).groupby("b")
    out = g.mean()
    res = pd.DataFrame({xcol: np.exp(out["lx"].to_numpy()), **{c: out[c].to_numpy() for c in ycols}})
    return res, g.size().to_numpy().astype(float)


def bin_uncertainty(data, xcol, ycols, n=MAX_POINTS):
    x = data[xcol].to_numpy(float)
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n + 1)
    counts = np.bincount(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1), minlength=n)
    for min_obs in (10, 3):
        if (counts >= min_obs).any():
            return _bin_uncertainty(data, xcol, ycols, n, min_obs)
    raise ValueError("no hay intervalos con 3 o más observaciones para estimar la incertidumbre")


def _bin_uncertainty(data, xcol, ycols, n, MIN_OBS_SE):
    """Error estándar RELATIVO del promedio de cada intervalo de log_bin (mismos intervalos), para pesos w = 1/SE_rel^2.
    1. Dentro de cada intervalo con >= 10 observaciones se quita la tendencia de la curva (recta en ln x) antes de medir
       la dispersión: en tiempos largos un intervalo abarca muchos segundos y la curva sube dentro de él; esa subida
       no es ruido.
    2. sigma_rel = desviación estándar de los residuos (m - 2 grados de libertad) / promedio del intervalo.
    3. El perfil de ruido se suaviza con la mediana de 5 intervalos vecinos (la resolución del equipo puede dar
       dispersión cero) y se interpola en ln x a los intervalos con menos de 10 observaciones (con 3 o 4 la desviación
       estándar tiene 1-2 grados de libertad y su mediana subestima el ruido a la mitad). Si ningún intervalo llega a
       10 observaciones se usan los de 3 o más.
    4. SE_rel = sigma_rel / sqrt(m_ef). Con adquisición rápida el equipo filtra la señal y el ruido de lecturas
       consecutivas está correlacionado: m observaciones valen menos que m independientes. Se mide la autocorrelación
       de lag 1 (rho) de los residuos dentro de los intervalos y se usa el número efectivo de un proceso AR(1),
       m_ef = m (1 - rho)/(1 + rho) (mínimo 1).
    Devuelve (array (intervalos, columnas) con SE_rel, array con sigma_rel por observación, lista de rho por columna)."""
    x = data[xcol].to_numpy(float)
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n + 1)
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)
    lx = np.log(x)
    bins = np.unique(b)
    se_all, sig_all, rho_all = [], [], []
    for c in ycols:
        y = data[c].to_numpy(float)
        num = den = 0.0
        sig = np.full(len(bins), np.nan)
        cnt = np.zeros(len(bins))
        cen = np.zeros(len(bins))
        for i, k in enumerate(bins):
            sel = b == k
            m = int(sel.sum())
            cnt[i], cen[i] = m, lx[sel].mean()
            if m >= MIN_OBS_SE:
                u = lx[sel] - lx[sel].mean()
                yy = y[sel]
                slope = float(u @ (yy - yy.mean()) / (u @ u)) if u @ u > 0 else 0.0
                r = yy - yy.mean() - slope * u
                sig[i] = float(np.sqrt(r @ r / (m - 2))) / abs(yy.mean())
                if m >= 10:   # autocorrelación de lag 1 dentro del intervalo (datos en orden de tiempo)
                    num += float(r[:-1] @ r[1:])
                    den += float(r @ r)
        ok = np.isfinite(sig)
        if not ok.any():
            raise ValueError("no hay intervalos con suficientes observaciones para estimar la incertidumbre")
        sm = sig.copy()
        idx = np.flatnonzero(ok)
        for j, i in enumerate(idx):   # mediana móvil sobre los intervalos con estimación
            sm[i] = float(np.median(sig[idx[max(0, j - 2):j + 3]]))
        sm = np.interp(cen, cen[ok], sm[ok])
        floor = max(1e-6, 0.01 * float(np.median(sm[ok])))   # la resolución del equipo puede dar dispersión ~0
        sm = np.maximum(sm, floor)
        rho = float(np.clip(num / den, 0.0, 0.99)) if den > 0 else 0.0
        m_ef = np.maximum(cnt * (1 - rho) / (1 + rho), 1.0)
        se_all.append(sm / np.sqrt(m_ef))
        sig_all.append(sm)
        rho_all.append(rho)
    return np.column_stack(se_all), np.column_stack(sig_all), rho_all


WEIGHTING = {
    "decada": "igual peso por década: cada intervalo logarítmico cuenta lo mismo, sin importar cuántas observaciones tenga",
    "todos": "todas las observaciones: cada intervalo pesa según cuántas observaciones promedia; equivale "
             "aproximadamente a ajustar cada fila con la pérdida de pyRheo",
    "incertidumbre": "incertidumbre de cada intervalo: w = 1/SE_rel^2, con SE_rel el error estándar relativo del promedio "
                     "(dispersión sin tendencia dentro del intervalo / sqrt(observaciones efectivas, corregidas por la "
                     "autocorrelación del ruido)); Singh y Ewoldt (2022). "
                     "Permite el chi^2 reducido como prueba de bondad de ajuste",
}
MAIN_FAMILY = ("SpringPot", "FractionalMaxwellGel", "FractionalKelvinVoigtS", "FractionalZenerSolidS")


def _bin_and_fit(test, data, x_col, ycols, weighting, models):
    """Promedia por intervalos, calcula los pesos y ajusta. Devuelve un dict con todo lo necesario para la respuesta."""
    n_raw = len(data)
    x_all = data[x_col].to_numpy(float)
    ys_all = [data[c].to_numpy(float) for c in ycols]
    binned, counts = log_bin(data, x_col, ycols)
    x = binned[x_col].to_numpy(float)
    ys = [binned[c].to_numpy(float) for c in ycols]
    pond = {"metodo": weighting}
    scale = None
    if weighting == "todos":
        w = counts / counts.mean()
    elif weighting == "incertidumbre":
        if len(x) == n_raw:
            w = np.ones_like(counts)
            pond["aviso"] = ("Los datos no se promediaron (hay pocas observaciones), así que no hay dispersión por intervalo "
                             "para estimar la incertidumbre: se usó igual peso.")
        else:
            try:
                se, sig, rho = bin_uncertainty(data, x_col, ycols)
            except ValueError as e:
                raise HTTPException(400, f"No se pudo estimar la incertidumbre por intervalo: {e}.")
            w_raw = np.concatenate([1 / se[:, j] ** 2 for j in range(se.shape[1])])
            scale = float(w_raw.mean())
            w = w_raw / scale
            pond.update({"se_rel_pct": (se * 100).T.tolist(), "ruido_por_observacion_pct": (sig * 100).T.tolist(),
                         "autocorrelacion_del_ruido": rho,
                         "nota": "SE relativo del promedio de cada intervalo; los pesos se normalizan a media 1 "
                                 "(no cambia el ajuste) y el chi^2 reducido usa la escala original"})
    else:
        w = np.ones_like(counts)
    results = run_fit(test, x, ys, models, w=w, x_all=x_all, ys_all=ys_all)
    if scale is not None:
        n_fit = len(np.concatenate(ys))
        for r in results:
            if "error" not in r:
                # chi^2 reducido = sum ((y - yhat)/y / SE_rel)^2 / (n - k): ~1 si el modelo reproduce los datos dentro del ruido
                r["chi2_red"] = float(r["rss_w"] * scale / max(n_fit - len(r["params"]), 1))
    return {"n_raw": n_raw, "x_all": x_all, "ys_all": ys_all, "x": x, "ys": ys, "counts": counts, "results": results,
            "ponderacion": pond}


def _param_table(r):
    """Parámetros con IC para comparar dos ajustes: notación del doctor si existe, si no los de pyRheo."""
    if r.get("doctor"):
        return {**r["doctor"]["parametros"], **(r["doctor"].get("derivados") or {})}
    return r.get("incertidumbre") or {}


def ramp_sensitivity(test, data, x_col, ycols, weighting, ramp, results, recommendation):
    """Prueba de sensibilidad a la rampa (Flory y McKenna; 'A note on ... ramp tests', 2000): la respuesta a la rampa y
    al escalón coinciden aproximadamente después de ~10 veces la duración de la rampa t1. Se repite el ajuste con los
    datos desde 10 t1 y se compara cada parámetro con el del ajuste principal.
    Criterio: la diferencia es significativa si |delta| > 1.96 sqrt(SE_principal^2 + SE_10t1^2) (módulos y tiempos en
    escala logarítmica). Es conservador: trata los dos ajustes como independientes aunque comparten datos. La prueba de
    Hausman para estimadores anidados (varianza SE_10t1^2 - SE_principal^2) no aplica porque al recortar los datos se
    vuelven a promediar en otros intervalos con otros pesos; con datos sintéticos exactos daba falsas alarmas en alpha.
    Con el criterio conservador no hubo falsas alarmas y un artefacto de 1 % en los primeros segundos se detecta con
    z ~ 4.5. Comparar solo con el IC del ajuste principal (lo que proponía el plan) rechazaría por azar aun sin sesgo,
    porque el segundo ajuste tiene más ruido; ese dato se reporta como informativo («dentro del IC principal»)."""
    t1 = float(ramp["duracion_s"])
    t_cut = 10 * t1
    sub = data[data[x_col] >= t_cut]
    info = {"t1_s": t1, "t_corte_s": t_cut,
            "criterio": "|diferencia| ≤ 1.96·√(SE_principal² + SE₁₀ₜ₁²) (módulos y tiempos en escala logarítmica)"}
    if len(sub) < 20 or np.log10(sub[x_col].max() / t_cut) < 0.7:
        info["realizado"] = False
        info["motivo"] = (f"Desde 10·t1 = {t_cut:.3g} s quedan menos de 20 observaciones o menos de 0.7 décadas: "
                          "no hay datos suficientes para la prueba.")
        return info
    ok = {r["model"]: r for r in results if "error" not in r}
    chosen = [m for m in ([recommendation["modelo"]] if recommendation else []) + list(MAIN_FAMILY) if m in ok]
    chosen = list(dict.fromkeys(chosen))[:4]
    fit2 = _bin_and_fit(test, sub, x_col, ycols, weighting, chosen)
    info.update({"realizado": True, "observaciones": fit2["n_raw"],
                 "ventana_s": [float(fit2["x_all"].min()), float(fit2["x_all"].max())], "modelos": {}})
    for r2 in fit2["results"]:
        if "error" in r2 or r2["model"] not in ok:
            continue
        a, b = _param_table(ok[r2["model"]]), _param_table(r2)
        comp, consistent, comparable = {}, True, 0
        for k, ua in a.items():
            ub = b.get(k)
            if ub is None:
                continue
            row = {"principal": ua["valor"], "desde_10t1": ub["valor"], "ic_principal": [ua.get("inf"), ua.get("sup")],
                   "ic_10t1": [ub.get("inf"), ub.get("sup")],
                   "cambio_pct": float((ub["valor"] / ua["valor"] - 1) * 100) if ua["valor"] else None}
            if ua.get("inf") is not None and ua.get("sup") is not None:
                row["dentro_ic_principal"] = bool(ua["inf"] <= ub["valor"] <= ua["sup"])
            sea, seb = ua.get("error_estandar"), ub.get("error_estandar")
            if ua.get("determinado") and ub.get("determinado") and sea and seb and ua["valor"] > 0 and ub["valor"] > 0:
                if k in ("alpha", "beta", "kappa"):
                    delta, va, vb = ub["valor"] - ua["valor"], sea, seb
                else:   # escala logarítmica: SE relativo
                    delta, va, vb = np.log(ub["valor"] / ua["valor"]), sea / ua["valor"], seb / ub["valor"]
                sd = float(np.sqrt(va ** 2 + vb ** 2))   # conservador: trata los dos ajustes como independientes
                row["z"] = float(abs(delta) / sd)
                row["consistente"] = bool(row["z"] <= 1.96)
                consistent &= row["consistente"]
                comparable += 1
            else:
                row["consistente"] = None     # no determinado en alguno de los dos ajustes: no se compara
            comp[k] = row
        if comparable == 0:
            concl = "Ningún parámetro está determinado en ambos ajustes: la prueba no aplica."
        elif consistent:
            concl = ("Las diferencias entre el ajuste principal y el de 10·t1 son compatibles con el ruido: la corrección de "
                     "rampa no cambia el resultado. Se reporta el ajuste principal.")
        else:
            concl = ("Algún parámetro cambia más de lo que explica el ruido al quitar los datos antes de 10·t1: el resultado "
                     "depende del tramo inicial. Reporta el ajuste desde 10·t1 (o explica por qué no) y menciona la diferencia.")
            lag = ok[r2["model"]].get("autocorrelacion_lag1")
            if lag is not None and lag > 0.5:
                concl += (f" Ojo: los residuos del ajuste principal tienen tendencia (ρ₁ = {lag:.2f}), así que el modelo no "
                          "reproduce exactamente la forma de la curva y sus IC solo reflejan el ruido. Al quitar el tramo "
                          "inicial cambia el ajuste aunque la rampa no sesgue: la diferencia mezcla el efecto de la rampa con "
                          "la falta de ajuste del modelo y con la dependencia de los parámetros de la ventana medida. "
                          "Compara el cambio relativo (%) con lo que es relevante para tu conclusión.")
        info["modelos"][r2["model"]] = {"parametros": comp, "consistente": bool(consistent) if comparable else None,
                                         "conclusion": concl, "rms_pct": (r2.get("todos") or {}).get("rms_pct")}
    return info


# ---------------------------------------------------------------- endpoints
@app.get("/api/status")
def status():
    # «Recomendado» prueba todos los modelos salvo el automático de pyRheo y el Zener fraccionario de 6 parámetros en
    # creep (su creep requiere inversión numérica de Laplace y suele exceder el límite de tiempo)
    rec = {t: [m for m in available_models(t) if m != "auto" and not (t == "creep" and m == "FractionalZener")]
           for t in ("creep", "relaxation", "saos", "steady")}
    return {"pyrheo": PYRHEO, "import_error": IMPORT_ERROR, "models": {t: available_models(t) for t in ("creep", "relaxation", "saos", "steady")},
            "recommend_models": rec}


@app.post("/api/preview")
def preview(file: UploadFile = File(...), sheet: str = Form(""), sample: str = Form("")):
    tb = load_table(file, file.file.read(), sheet, sample)
    df = tb["df"]
    head = df.head(8).astype(object).where(df.head(8).notna(), None)
    guess = guess_fime(df) if tb["fime"] else guess_columns([str(c) for c in df.columns])
    return {"columns": [str(c) for c in df.columns], "rows": len(df), "head": head.values.tolist(),
            "sheets": tb["sheets"], "sheet": tb["sheet"], "samples": tb["samples"], "sample": tb["sample"],
            "format": "FIME" if tb["fime"] else "", "guess": guess}


@app.post("/api/fit")
def fit(file: UploadFile = File(...), test: str = Form(...), x_col: str = Form(...),
              y_col: str = Form(...), y2_col: str = Form(""), models: str = Form(...),
              sheet: str = Form(""), sample: str = Form(""), trim: str = Form("0"), weighting: str = Form("decada"),
              tolerancia: str = Form("1"), sensitivity: str = Form("1")):
    if test not in ("creep", "relaxation", "saos", "steady"):
        raise HTTPException(400, "Tipo de ensayo no válido.")
    if weighting not in WEIGHTING:
        raise HTTPException(400, "Ponderación no válida.")
    try:
        tol = float(tolerancia)
    except ValueError:
        tol = float("nan")
    if not (0 < tol <= 50):
        raise HTTPException(400, "La incertidumbre de medición debe estar entre 0 y 50 %.")
    tb = load_table(file, file.file.read(), sheet, sample)
    df = tb["df"]
    ramp = None
    if tb["fime"] and test in ("creep", "relaxation") and trim == "1":
        df, ramp = fime_segment(df, test)
    cols = [x_col, y_col] + ([y2_col] if test == "saos" else [])
    if test == "saos" and not y2_col:
        raise HTTPException(400, "SAOS necesita G′ y G″.")
    try:
        data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    except KeyError:
        raise HTTPException(400, "Alguna columna elegida no existe en el archivo.")
    data = data[(data > 0).all(axis=1)].sort_values(x_col)  # el ajuste ponderado y los ejes log requieren valores > 0
    if not tb["fime"] and test == "relaxation" and trim == "1" and len(data):
        data = data.iloc[int(np.argmax(data[y_col].to_numpy())):].copy()  # desde el pico, quita la rampa de carga
        data[x_col] = data[x_col] - data[x_col].iloc[0]
        data = data[data[x_col] > 0]
    if len(data) < 5:
        raise HTTPException(400, "Quedaron menos de 5 puntos válidos (positivos y numéricos).")
    F = _bin_and_fit(test, data, x_col, cols[1:], weighting, json.loads(models))
    n_raw, x_all, x, ys, counts, results = F["n_raw"], F["x_all"], F["x"], F["ys"], F["counts"], F["results"]
    recommendation = recommend(results, len(np.concatenate(ys)), tol)
    sens = None
    if ramp and sensitivity == "1":
        sens = ramp_sensitivity(test, data, x_col, cols[1:], weighting, ramp, results, recommendation)

    decades = float(np.log10(x_all.max() / x_all.min())) if len(x_all) > 1 else 0.0
    reduction = (f"Se promediaron las {n_raw} observaciones en {len(x)} intervalos logarítmicos "
                 f"(~{len(x) / decades:.0f} por década); " if len(x) < n_raw else f"Se usaron las {n_raw} observaciones; ")
    resp = {"n": len(x), "n_raw": n_raw, "x": x.tolist(), "ys": [y.tolist() for y in ys], "counts": counts.tolist(),
            "results": results, "pyrheo": PYRHEO, "weighting": weighting, "recomendacion": recommendation,
            "sensibilidad_rampa": sens,
            "preprocesamiento": {
                "rampa": ramp, "origen_del_tiempo": ("mitad de la rampa, t - t1/2 (Zapas-Craft; superposición de Boltzmann)"
                                                      if ramp else "el del archivo"),
                "datos_desde": "fin de la rampa" if ramp else "inicio del archivo",
                "ventana_s": [float(x_all.min()), float(x_all.max())], "observaciones": n_raw, "intervalos": len(x),
                "filtro": "solo valores numéricos y positivos", "ponderacion": F["ponderacion"]},
            "settings": {**fit_settings(test), "intervalos_logaritmicos": len(x), "observaciones": n_raw,
                         "reduccion": "promedio de todas las observaciones dentro de cada intervalo de igual ancho en log10(x)",
                         "ponderacion": WEIGHTING[weighting], "incertidumbre_medicion_pct": tol, "filtro": "solo valores positivos",
                         "metricas_todos": "RMS relativo y percentil 95 del error relativo absoluto sobre TODAS las observaciones válidas"}}
    weight_txt = {"decada": "ponderación: igual peso por década. ",
                  "todos": "ponderación: todas las observaciones (cada intervalo pesa según cuántas promedia). ",
                  "incertidumbre": "ponderación: 1/SE² de cada intervalo (incertidumbre del promedio). "}[weighting]
    if tb["fime"] and test in ("creep", "relaxation"):
        resp["modo"] = "tension"
        resp["axes"] = ["t [s]", "D(t) [1/MPa]" if test == "creep" else "E(t) [MPa]"]
        resp["rampa"] = ramp
        resp["note"] = (f"Muestra {tb['sample']}. " +
                        (f"Rampa de carga de {ramp['duracion_s']:.3g} s: el tiempo se mide desde la mitad de la rampa "
                         "(superposición de Boltzmann) y se usan los datos desde su final. " if ramp else
                         "Sin corrección de rampa: los primeros segundos están sesgados por la carga. ") +
                        ("D(t) = Deformación/Tensión, complianza de creep en tensión, en 1/MPa. " if test == "creep" else
                         "E(t) = Tensión/Deformación, módulo de relajación en tensión, en MPa. ") +
                        "pyRheo llama G a sus parámetros de módulo; aquí son módulos de tensión. " +
                        reduction + weight_txt)
        # la deformación de los archivos FIME viene del desplazamiento del cabezal, no de un extensómetro
        resp["aviso_deformacion"] = ("Deformación nominal: calculada con el desplazamiento del cabezal (separación de mordazas), "
                                     "no con extensómetro. D(t) y E(t) son APARENTES: incluyen la deformación en las mordazas "
                                     "y cualquier deslizamiento. ISO 899-1 exige extensómetro para datos de diseño y acepta la "
                                     "separación de mordazas para investigación o control de calidad. Repórtalo como limitación.")
    else:
        # archivos genéricos: no se suponen unidades; los ejes llevan el nombre de las columnas
        resp["axes"] = [x_col, f"{y_col} / {y2_col}" if test == "saos" else y_col]
        resp["note"] = reduction + weight_txt
    return resp


@app.post("/api/tensile")
def tensile(file: UploadFile = File(...), sheet: str = Form(""), x_col: str = Form(""), y_col: str = Form(""),
                  secant: float = Form(1.0), strain_unit: str = Form("pct"), stress_unit: str = Form("MPa")):
    """Ensayo de tensión analizado según ASTM D638-14 (ver tensile.py). No usa pyRheo.
    Internamente todo se calcula con deformación en % y esfuerzo en MPa; los archivos FIME ya vienen así y en
    archivos genéricos se convierten según las unidades que indique el usuario."""
    if not (0 < secant <= 100):
        raise HTTPException(400, "La deformación designada para el módulo secante debe estar entre 0 y 100 %.")
    STRAIN = {"pct": 1.0, "frac": 100.0}
    STRESS = {"MPa": 1.0, "kPa": 1e-3, "Pa": 1e-6}
    if strain_unit not in STRAIN or stress_unit not in STRESS:
        raise HTTPException(400, "Unidades no válidas.")
    tb = load_table(file, file.file.read(), sheet, "")
    series = {}
    if tb["fime"]:
        for label, d in tb["all"].items():
            m = fime_meta(d)
            col = lambda p: next((c for c in d.columns if _norm(c).startswith(p)), None)
            fcol, xcol = col("fuerza"), col("desplazamiento")
            dd = d.dropna(subset=[m["e"], m["s"]])
            series[label] = dict(strain_pct=dd[m["e"]], stress=dd[m["s"]], time=dd[m["t"]],
                                 force=dd[fcol] if fcol else None, disp=dd[xcol] if xcol else None)
    else:
        try:
            dd = tb["df"][[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        except KeyError:
            raise HTTPException(400, "Elige las columnas de deformación y esfuerzo.")
        series["M1"] = dict(strain_pct=dd[x_col] * STRAIN[strain_unit], stress=dd[y_col] * STRESS[stress_unit])
    out, errors = [], []
    for label, kw in series.items():
        kw = {k: (None if v is None else pd.to_numeric(v, errors="coerce").to_numpy(float)) for k, v in kw.items()}
        try:
            r = tensile_d638.analyze(secant_pct=secant, **kw)
        except Exception as e:
            errors.append({"sample": label, "error": str(e)})
            continue
        # claves cortas usadas por la gráfica, el CSV y el código Python
        r.update(sample=label, strain=r["deformacion"], stress=r["esfuerzo"], E=r["modulo"]["valor"],
                 sigma_max=r["resistencia_tension"], eps_break=r["ruptura"]["deformacion"])
        out.append(r)
    if not out:
        raise HTTPException(400, "No encontré datos suficientes de esfuerzo y deformación." +
                            (" " + "; ".join(f"{e['sample']}: {e['error']}" for e in errors) if errors else ""))
    checks = tensile_d638.checks(out)
    if not tb["fime"]:
        emax = max(max(r["deformacion"]) for r in out)
        checks.insert(0, {"clausula": "Unidades", "estado": "aviso" if (strain_unit == "pct" and emax < 1) else "info",
                          "texto": f"Unidades de entrada: deformación en {'%' if strain_unit == 'pct' else 'fracción (mm/mm)'}, esfuerzo en "
                                   f"{stress_unit}; los resultados se reportan en % y MPa." +
                                   (f" La deformación máxima es {emax:.3g} %: si tus datos están en mm/mm, elige «fracción»."
                                    if strain_unit == "pct" and emax < 1 else "")})
    return {"results": out, "errors": errors, "format": "FIME" if tb["fime"] else "", "norma": tensile_d638.NORMA,
            "resumen": tensile_d638.summarize(out), "verificaciones": checks,
            "implementacion": tensile_d638.IMPLEMENTACION, "secante_designada": secant,
            "unidades_entrada": None if tb["fime"] else {"deformacion": strain_unit, "esfuerzo": stress_unit}}


class PredictIn(BaseModel):
    model: str
    params: Dict[str, float]
    target: str            # "saos" | "relaxation" | "creep"
    x_min: float           # 10^x_min ... 10^x_max (rad/s para saos, s para las otras)
    x_max: float
    n: int = 200


@app.post("/api/predict")
def predict(q: PredictIn):
    """Usa los parámetros de un modelo ya ajustado para calcular otra respuesta (p. ej. G′, G″ a partir de creep)."""
    if q.target not in ("saos", "relaxation", "creep"):
        raise HTTPException(400, "Respuesta no válida.")
    if not (-12 <= q.x_min < q.x_max <= 12) or not (10 <= q.n <= 1000):
        raise HTTPException(400, "Revisa el rango: mínimo < máximo, entre 10^-12 y 10^12, con 10 a 1000 puntos.")
    x = np.logspace(q.x_min, q.x_max, q.n)
    try:
        ys = predict_response(q.model, q.params, q.target, x)
    except Exception as e:
        raise HTTPException(400, str(e))
    clean = lambda v: [float(a) if np.isfinite(a) and a > 0 else None for a in v]
    return {"x": x.tolist(), "ys": [clean(y) for y in ys], "target": q.target, "model": q.model}


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")


app.mount("/static", StaticFiles(directory=FRONT), name="static")
