"""Inferencia del clasificador de intención (CU-31/CU-46).

Carga el modelo TensorFlow entrenado y reconstruye el vectorizador desde el
vocabulario guardado. TensorFlow se importa de forma PEREZOSA: así el
microservicio arranca aunque TF no esté instalado (el endpoint responde 503 y el
backend Spring cae a su KB por palabras).
"""
import json
import os
from functools import lru_cache

AQUI = os.path.dirname(__file__)
RUTA_MODELO = os.path.join(AQUI, "modelo_intencion.keras")
RUTA_VOCAB = os.path.join(AQUI, "vocab.json")
RUTA_ETIQUETAS = os.path.join(AQUI, "etiquetas.json")

MAX_TOKENS = 2000
LARGO_SEC = 24


def disponible() -> bool:
    """¿Está el modelo entrenado en disco? (no importa TF)."""
    return os.path.exists(RUTA_MODELO) and os.path.exists(RUTA_VOCAB)


@lru_cache(maxsize=1)
def _cargar():
    # TF se importa AQUÍ (lazy). Si no está instalado, lanza ImportError y el
    # endpoint lo traduce a 503.
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers

    modelo = tf.keras.models.load_model(RUTA_MODELO)
    with open(RUTA_VOCAB, encoding="utf-8") as fh:
        vocab = json.load(fh)
    with open(RUTA_ETIQUETAS, encoding="utf-8") as fh:
        etiquetas = json.load(fh)

    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=LARGO_SEC,
    )
    vectorizer.set_vocabulary(vocab)

    def _clasificar(texto: str):
        x = vectorizer(tf.constant([texto or ""]))
        probs = modelo.predict(x, verbose=0)[0]
        i = int(np.argmax(probs))
        return etiquetas[i], float(probs[i])

    return _clasificar


def clasificar(texto: str) -> tuple[str, float]:
    """Devuelve (intencion, confianza) usando el modelo TF."""
    return _cargar()(texto)
