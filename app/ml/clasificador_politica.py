"""Clasificador de política (CU-40) — modelo TensorFlow PROPIO y REENTRENABLE.

Arquitectura idéntica al clasificador de intención (TextVectorization +
Embedding + GlobalAveragePooling + Dense→softmax), pero sus etiquetas son los
IDs de las POLÍTICAS activas y se RE-ENTRENA cuando cambian (P2 §3.2.2). No es
una API externa: es un modelo entrenado con TensorFlow, demostrable en vivo.

- ``entrenar(politicas)``  → genera el dataset y entrena (lo llama el backend
  vía /asignacion/reentrenar al crear/activar políticas).
- ``clasificar_sobre(descripcion, activas)`` → predice y RESTRINGE el resultado
  a las políticas activas que manda el backend (renormaliza). Si el modelo no
  está entrenado o no conoce ninguna de las activas, devuelve None para que el
  router caiga a la heurística (nunca rompe el flujo CU-40).

TensorFlow se importa de forma PEREZOSA: el microservicio arranca aunque TF no
esté. El modelo en disco se recarga solo si su mtime cambió (tras un reentreno).
"""
import json
import os
import threading

AQUI = os.path.dirname(__file__)
RUTA_MODELO = os.path.join(AQUI, "modelo_politica.keras")
RUTA_VOCAB = os.path.join(AQUI, "vocab_politica.json")
RUTA_ETIQUETAS = os.path.join(AQUI, "etiquetas_politica.json")
RUTA_NOMBRES = os.path.join(AQUI, "nombres_politica.json")

MAX_TOKENS = 2000
LARGO_SEC = 24
DIM_EMB = 32
EPOCHS = 140

_lock = threading.Lock()
_estado: dict = {"mtime": None, "fn": None, "etiquetas": []}


def disponible() -> bool:
    """¿Hay un modelo de política entrenado en disco? (no importa TF)."""
    return os.path.exists(RUTA_MODELO) and os.path.exists(RUTA_VOCAB) and os.path.exists(RUTA_ETIQUETAS)


def _recargar_si_cambio() -> None:
    """Carga (o recarga) el modelo si el archivo cambió. Thread-safe y perezoso
    en TF: si TF/el modelo no están, deja fn=None y el caller cae a heurística."""
    if not disponible():
        _estado["fn"] = None
        return
    mtime = os.path.getmtime(RUTA_MODELO)
    if _estado["fn"] is not None and _estado["mtime"] == mtime:
        return
    with _lock:
        if _estado["fn"] is not None and _estado["mtime"] == mtime:
            return
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras import layers

        modelo = tf.keras.models.load_model(RUTA_MODELO)
        with open(RUTA_VOCAB, encoding="utf-8") as fh:
            vocab = json.load(fh)
        with open(RUTA_ETIQUETAS, encoding="utf-8") as fh:
            etiquetas = json.load(fh)

        vectorizer = layers.TextVectorization(
            max_tokens=MAX_TOKENS, output_mode="int", output_sequence_length=LARGO_SEC,
        )
        vectorizer.set_vocabulary(vocab)

        def _probs(texto: str):
            x = vectorizer(tf.constant([texto or ""]))
            return modelo.predict(x, verbose=0)[0]

        _estado.update(mtime=mtime, fn=_probs, etiquetas=etiquetas)


def clasificar_sobre(descripcion: str, activas: list) -> tuple[str, float, list[dict]] | None:
    """Predice la política y restringe a las activas que manda el backend.

    ``activas``: lista de objetos/dicts con .id/.nombre (PoliticaActiva). Devuelve
    (politica_id, confianza, top3) o None si el modelo no puede ayudar."""
    if not descripcion or not descripcion.strip() or not activas:
        return None
    try:
        _recargar_si_cambio()
    except Exception:  # noqa: BLE001 — TF no instalado / modelo ilegible → heurística
        return None
    fn = _estado["fn"]
    if fn is None:
        return None

    etiquetas = _estado["etiquetas"]
    probs = fn(descripcion)
    id2prob = {etiquetas[i]: float(probs[i]) for i in range(len(etiquetas))}

    pares = []
    for p in activas:
        pid = getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else None)
        nombre = getattr(p, "nombre", None) or (p.get("nombre") if isinstance(p, dict) else "")
        if pid:
            pares.append((str(pid), str(nombre or ""), id2prob.get(str(pid), 0.0)))

    suma = sum(prob for _, _, prob in pares)
    if suma <= 1e-6:
        # El modelo no conoce ninguna de las políticas activas (está desfasado) →
        # que el router use la heurística y dispare un reentrenamiento.
        return None

    pares.sort(key=lambda x: x[2], reverse=True)
    top3 = [{"politica_id": pid, "nombre": nom, "confianza": round(prob / suma, 3)}
            for pid, nom, prob in pares[:3]]
    return top3[0]["politica_id"], top3[0]["confianza"], top3


def entrenar(politicas: list[dict]) -> dict:
    """Genera el dataset desde las políticas y entrena el modelo TF. Requiere ≥2
    políticas (un clasificador necesita al menos 2 clases). Devuelve estadísticas."""
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models

    from app.ml.dataset_politica import generar_dataset

    textos, labels = generar_dataset(politicas)
    etiquetas = sorted(set(labels))  # ids de política, índice estable
    if len(etiquetas) < 2:
        return {"entrenado": False, "motivo": "se requieren al menos 2 políticas activas",
                "politicas": len(etiquetas)}

    idx = {pid: i for i, pid in enumerate(etiquetas)}
    y = np.array([idx[l] for l in labels], dtype="int32")

    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_mode="int", output_sequence_length=LARGO_SEC,
    )
    vectorizer.adapt(textos)
    X = vectorizer(tf.constant(textos)).numpy()

    modelo = models.Sequential([
        tf.keras.Input(shape=(LARGO_SEC,), dtype="int32"),
        # mask_zero: el padding (índice 0) no diluye el promedio → mejor en
        # descripciones cortas, que es justo lo que dicta el cliente.
        layers.Embedding(MAX_TOKENS, DIM_EMB, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(etiquetas), activation="softmax"),
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = modelo.fit(X, y, epochs=EPOCHS, verbose=0)

    modelo.save(RUTA_MODELO)
    vocab = vectorizer.get_vocabulary(include_special_tokens=False)
    with open(RUTA_VOCAB, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)
    with open(RUTA_ETIQUETAS, "w", encoding="utf-8") as fh:
        json.dump(etiquetas, fh, ensure_ascii=False)
    # id → nombre (conveniencia para logs/depuración; la inferencia usa el del request).
    nombres = {str(p.get("id")): str(p.get("nombre") or "") for p in politicas if p.get("id")}
    with open(RUTA_NOMBRES, "w", encoding="utf-8") as fh:
        json.dump(nombres, fh, ensure_ascii=False)

    # Forzar recarga del modelo en memoria en la próxima inferencia.
    _estado["mtime"] = None
    _estado["fn"] = None

    return {
        "entrenado": True,
        "politicas": len(etiquetas),
        "frases": len(textos),
        "accuracy": round(float(hist.history["accuracy"][-1]), 4),
    }


if __name__ == "__main__":
    # Entrenamiento de demo con las políticas semilla + prueba con frases reales.
    from app.ml.dataset_politica import SEED_POLITICAS

    print("Entrenando clasificador de política (demo, políticas semilla)…")
    stats = entrenar(SEED_POLITICAS)
    print("Stats:", stats)

    pruebas = [
        "quiero poner luz en mi casa que estoy construyendo",
        "me cortaron la electricidad porque no pague y ya pague",
        "compre una casa y quiero el medidor a mi nombre",
        "necesito una nueva conexion electrica para mi negocio",
        "no tengo servicio porque debia y ya cancele la deuda",
    ]
    print("\n=== prueba ===")
    for t in pruebas:
        r = clasificar_sobre(t, SEED_POLITICAS)
        print(f"  '{t}'\n    -> {r[0] if r else None} ({r[1] if r else '-'})  top3={[c['politica_id'] for c in r[2]] if r else []}")
