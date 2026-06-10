"""Motor de enrutamiento — modelos TensorFlow PROPIOS (CU-42/43/44/45).

DOMINIO-AGNÓSTICO: los modelos usan SOLO señales operativas genéricas (carga del
departamento, complejidad, hora, tiempo de espera, longitud/repeticiones de la
secuencia, deltas de tiempo). Nada de vocabulario de un negocio concreto, así que
sirven igual para una eléctrica, una clínica o un municipio.

Como aún no hay histórico real, los modelos se ENTRENAN con datos SINTÉTICOS
generados de una función latente plausible: el modelo aprende una función no
lineal real (no es una fórmula a mano) y es la base que luego se RE-ENTRENA con
métricas reales cuando se acumulan (P2 §3.2.4 + reentrenamiento §3.2.2).

Modelos:
  - riesgo     : clasificador  → prob. de superar el SLA (CU-43).
  - prioridad  : regresor      → score de urgencia para ordenar la bandeja (CU-44).
  - anomalia   : autoencoder   → score por error de reconstrucción (CU-45).
  - ruta       : clasificador  → necesidad de cada nodo; omite opcionales (CU-42).

TF se importa de forma perezosa; cada modelo se recarga si su archivo cambió.
"""
import json
import math
import os

AQUI = os.path.dirname(__file__)
RUTA_RIESGO = os.path.join(AQUI, "modelo_riesgo.keras")
RUTA_PRIORIDAD = os.path.join(AQUI, "modelo_prioridad.keras")
RUTA_ANOMALIA = os.path.join(AQUI, "modelo_anomalia.keras")
RUTA_RUTA = os.path.join(AQUI, "modelo_ruta.keras")
RUTA_META = os.path.join(AQUI, "enrutamiento_meta.json")

_SEED = 42
_cache: dict = {}


# ════════════════════════════════════════════════════════════════════════════
#  Feature engineering (compartido train/inferencia) — todo genérico
# ════════════════════════════════════════════════════════════════════════════
def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def feat_riesgo(carga: float, complejidad: float, hora: int, dia: int) -> list[float]:
    """[carga, complejidad, sin(hora), cos(hora), finde]."""
    h = (hora or 0) % 24
    return [
        _clip01(carga), _clip01(complejidad),
        math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
        1.0 if (dia or 0) in (6, 7) else 0.0,
    ]


def feat_prioridad(riesgo_score: float, prioridad_manual: int, espera_dias: float) -> list[float]:
    """[riesgo, prioridad_norm, espera_norm]."""
    prio = (min(max(prioridad_manual or 3, 1), 3) - 1) / 2.0
    return [_clip01(riesgo_score), prio, _clip01((espera_dias or 0) / 14.0)]


def feat_secuencia(transiciones: list[dict]) -> list[float]:
    """Rasgos genéricos de una secuencia de trámite (para el autoencoder)."""
    n = len(transiciones)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    nodos = [t.get("nodo", "") for t in transiciones]
    distintos = len(set(nodos))
    from collections import Counter
    max_rep = max(Counter(nodos).values())
    deltas = [float(t.get("delta_segundos", 0) or 0) for t in transiciones]
    max_d = max(deltas) if deltas else 0.0
    mean_d = (sum(deltas) / len(deltas)) if deltas else 0.0
    return [
        _clip01(n / 12.0),                       # longitud
        _clip01((n - distintos) / n),            # ratio de repetición (loops)
        _clip01((max_rep - 1) / 4.0),            # repetición máxima de un nodo
        _clip01(max_d / (3 * 86400.0)),          # delta máximo (cap 3 días)
        _clip01(mean_d / 86400.0),               # delta medio (cap 1 día)
    ]


def feat_nodo(tipo: str, opcional: bool, orden: int, total: int) -> list[float]:
    """[es_actividad, opcional, orden_norm]."""
    return [
        1.0 if (tipo or "").lower() == "actividad" else 0.0,
        1.0 if opcional else 0.0,
        _clip01(orden / max(total, 1)),
    ]


# ════════════════════════════════════════════════════════════════════════════
#  Entrenamiento (datos sintéticos, función latente plausible)
# ════════════════════════════════════════════════════════════════════════════
def entrenar_todos(n: int = 5000) -> dict:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models

    rng = np.random.default_rng(_SEED)
    tf.random.set_seed(_SEED)
    stats: dict = {}

    # ── CU-43 riesgo (clasificación: ¿superará el SLA?) ──
    carga = rng.uniform(0, 1, n); compl = rng.uniform(0, 1, n)
    hora = rng.integers(0, 24, n); dia = rng.integers(1, 8, n)
    z = (-1.6 + 2.3 * carga + 1.9 * compl + 1.0 * carga * compl
         + 0.7 * (hora >= 15) + 0.5 * np.isin(dia, (6, 7)))
    p = 1 / (1 + np.exp(-(z + rng.normal(0, 0.15, n))))
    y = (rng.uniform(0, 1, n) < p).astype("float32")
    Xr = np.array([feat_riesgo(c, k, int(h), int(d))
                   for c, k, h, d in zip(carga, compl, hora, dia)], dtype="float32")
    m_riesgo = models.Sequential([
        tf.keras.Input(shape=(5,)),
        layers.Dense(16, activation="relu"), layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    m_riesgo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    h_r = m_riesgo.fit(Xr, y, epochs=40, batch_size=64, verbose=0)
    m_riesgo.save(RUTA_RIESGO)
    stats["riesgo"] = {"accuracy": round(float(h_r.history["accuracy"][-1]), 4)}

    # ── CU-44 prioridad (regresión: urgencia 0..1) ──
    rs = rng.uniform(0, 1, n); pm = rng.integers(1, 4, n); esp = rng.uniform(0, 14, n)
    prio = (pm - 1) / 2.0
    urg = 1 / (1 + np.exp(-(-0.6 + 1.8 * rs + 1.3 * prio + 1.1 * (esp / 14.0)
                            + rng.normal(0, 0.1, n))))
    Xp = np.array([feat_prioridad(a, int(b), c) for a, b, c in zip(rs, pm, esp)], dtype="float32")
    m_prio = models.Sequential([
        tf.keras.Input(shape=(3,)),
        layers.Dense(12, activation="relu"), layers.Dense(1, activation="sigmoid"),
    ])
    m_prio.compile(optimizer="adam", loss="mse", metrics=["mae"])
    h_p = m_prio.fit(Xp, urg.astype("float32"), epochs=60, batch_size=64, verbose=0)
    m_prio.save(RUTA_PRIORIDAD)
    stats["prioridad"] = {"mae": round(float(h_p.history["mae"][-1]), 4)}

    # ── CU-45 anomalía (autoencoder sobre secuencias NORMALES) ──
    nn = n
    largo = rng.integers(2, 7, nn)
    Xa = []
    for L in largo:
        # secuencia "normal": sin loops, deltas pequeños (minutos..pocas horas)
        trans = [{"nodo": f"n{i}", "delta_segundos": float(rng.uniform(300, 6 * 3600))}
                 for i in range(int(L))]
        Xa.append(feat_secuencia(trans))
    Xa = np.array(Xa, dtype="float32")
    ae = models.Sequential([
        tf.keras.Input(shape=(5,)),
        layers.Dense(4, activation="relu"), layers.Dense(2, activation="relu"),
        layers.Dense(4, activation="relu"), layers.Dense(5, activation="linear"),
    ])
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(Xa, Xa, epochs=60, batch_size=64, verbose=0)
    ae.save(RUTA_ANOMALIA)
    recon = ae.predict(Xa, verbose=0)
    err = np.mean((Xa - recon) ** 2, axis=1)
    # Umbral con MARGEN: la separación normal↔anomalía es enorme (normal ~1e-3,
    # anomalía >0.3), así que un umbral holgado evita falsos positivos sin perder
    # las anomalías reales.
    umbral = float(max(np.percentile(err, 99.5) * 3.0, 0.01))
    stats["anomalia"] = {"umbral": round(umbral, 6)}

    # ── CU-42 ruta (necesidad de cada nodo) ──
    es_act = rng.integers(0, 2, n); opc = rng.integers(0, 2, n); orden = rng.uniform(0, 1, n)
    # Necesario casi siempre; un nodo OPCIONAL de tipo actividad puede ser omitible.
    zr = 3.0 - 3.2 * (opc * es_act) + rng.normal(0, 0.3, n)
    yr = (1 / (1 + np.exp(-zr)) > 0.5).astype("float32")
    Xn = np.array([feat_nodo("actividad" if e else "inicio", bool(o), int(r * 10), 10)
                   for e, o, r in zip(es_act, opc, orden)], dtype="float32")
    m_ruta = models.Sequential([
        tf.keras.Input(shape=(3,)),
        layers.Dense(8, activation="relu"), layers.Dense(1, activation="sigmoid"),
    ])
    m_ruta.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    h_n = m_ruta.fit(Xn, yr, epochs=40, batch_size=64, verbose=0)
    m_ruta.save(RUTA_RUTA)
    stats["ruta"] = {"accuracy": round(float(h_n.history["accuracy"][-1]), 4)}

    with open(RUTA_META, "w", encoding="utf-8") as fh:
        json.dump({"umbral_anomalia": umbral}, fh)

    _cache.clear()
    return stats


# ════════════════════════════════════════════════════════════════════════════
#  Carga perezosa (con recarga por mtime) + inferencia
# ════════════════════════════════════════════════════════════════════════════
def _modelo(ruta: str):
    """Carga un .keras (perezoso, recarga si cambió el archivo). None si no existe."""
    if not os.path.exists(ruta):
        return None
    mtime = os.path.getmtime(ruta)
    cached = _cache.get(ruta)
    if cached and cached[0] == mtime:
        return cached[1]
    import tensorflow as tf
    modelo = tf.keras.models.load_model(ruta)
    _cache[ruta] = (mtime, modelo)
    return modelo


def _meta() -> dict:
    if os.path.exists(RUTA_META):
        with open(RUTA_META, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def disponible() -> bool:
    return all(os.path.exists(r) for r in (RUTA_RIESGO, RUTA_PRIORIDAD, RUTA_ANOMALIA, RUTA_RUTA))


def predecir_riesgo(carga: float, complejidad: float, hora: int, dia: int) -> float | None:
    m = _modelo(RUTA_RIESGO)
    if m is None:
        return None
    import numpy as np
    x = np.array([feat_riesgo(carga, complejidad, hora, dia)], dtype="float32")
    return float(m.predict(x, verbose=0)[0][0])


def predecir_prioridad(riesgo_score: float, prioridad_manual: int, espera_dias: float) -> float | None:
    m = _modelo(RUTA_PRIORIDAD)
    if m is None:
        return None
    import numpy as np
    x = np.array([feat_prioridad(riesgo_score, prioridad_manual, espera_dias)], dtype="float32")
    return float(m.predict(x, verbose=0)[0][0])


def score_anomalia(transiciones: list[dict]) -> tuple[float, float] | None:
    """Devuelve (error_reconstruccion, umbral) o None si no hay modelo."""
    m = _modelo(RUTA_ANOMALIA)
    if m is None:
        return None
    import numpy as np
    x = np.array([feat_secuencia(transiciones)], dtype="float32")
    recon = m.predict(x, verbose=0)
    err = float(np.mean((x - recon) ** 2))
    umbral = float(_meta().get("umbral_anomalia", 0.02))
    return err, umbral


def predecir_necesidad_nodo(tipo: str, opcional: bool, orden: int, total: int) -> float | None:
    m = _modelo(RUTA_RUTA)
    if m is None:
        return None
    import numpy as np
    x = np.array([feat_nodo(tipo, opcional, orden, total)], dtype="float32")
    return float(m.predict(x, verbose=0)[0][0])


if __name__ == "__main__":
    print("Entrenando modelos de enrutamiento (sintético, dominio-agnóstico)…")
    print("Stats:", entrenar_todos())

    print("\n=== riesgo (carga, compl, hora, dia) ===")
    for c, k, h, d in [(0.9, 0.8, 18, 6), (0.2, 0.1, 10, 2), (0.6, 0.5, 12, 3)]:
        print(f"  carga={c} compl={k} h={h} d={d} -> prob_sla={predecir_riesgo(c, k, h, d):.3f}")

    print("=== prioridad (riesgo, prio_manual, espera_dias) ===")
    for r, pm, e in [(0.9, 3, 10), (0.2, 1, 1), (0.5, 2, 5)]:
        print(f"  riesgo={r} prio={pm} espera={e} -> urgencia={predecir_prioridad(r, pm, e):.3f}")

    print("=== anomalía (err vs umbral) ===")
    normal = [{"nodo": f"n{i}", "delta_segundos": 3600} for i in range(4)]
    loop = [{"nodo": "n1", "delta_segundos": 3600} for _ in range(6)]
    lento = [{"nodo": f"n{i}", "delta_segundos": 4 * 86400} for i in range(3)]
    for nombre, sec in [("normal", normal), ("loop", loop), ("lento", lento)]:
        err, umb = score_anomalia(sec)
        print(f"  {nombre:7} err={err:.5f} umbral={umb:.5f} -> {'ANOMALÍA' if err > umb else 'ok'}")

    print("=== ruta (necesidad de nodo) ===")
    for tipo, opc in [("actividad", False), ("actividad", True), ("inicio", False)]:
        print(f"  tipo={tipo:9} opcional={opc} -> necesidad={predecir_necesidad_nodo(tipo, opc, 3, 6):.3f}")
