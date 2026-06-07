"""Entrena el clasificador de intención del asistente (TensorFlow/Keras) — CU-31/CU-46.

Modelo: TextVectorization (vocabulario) + Embedding + GlobalAveragePooling +
Dense → softmax sobre las intenciones. Es un MODELO PROPIO entrenado con
TensorFlow (no una API externa), demostrable en vivo.

Para robustez entre versiones de TF, entrenamos un modelo de ENTEROS y guardamos
el vocabulario aparte (la capa de texto se reconstruye en inferencia).

Correr:  cd ia_service && .venv312/Scripts/python -m app.ml.entrenar
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.ml.dataset import INTENCIONES, ETIQUETAS

AQUI = os.path.dirname(__file__)
RUTA_MODELO = os.path.join(AQUI, "modelo_intencion.keras")
RUTA_VOCAB = os.path.join(AQUI, "vocab.json")
RUTA_ETIQUETAS = os.path.join(AQUI, "etiquetas.json")

MAX_TOKENS = 2000
LARGO_SEC = 24
DIM_EMB = 32


def entrenar() -> None:
    textos: list[str] = []
    y: list[int] = []
    for intent, frases in INTENCIONES.items():
        idx = ETIQUETAS.index(intent)
        for f in frases:
            textos.append(f)
            y.append(idx)
    y = np.array(y, dtype="int32")

    # Vectorizador de texto (palabra -> índice).
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=LARGO_SEC,
    )
    vectorizer.adapt(textos)

    # Texto -> enteros (lo que come el modelo).
    X = vectorizer(tf.constant(textos)).numpy()

    modelo = models.Sequential([
        tf.keras.Input(shape=(LARGO_SEC,), dtype="int32"),
        layers.Embedding(MAX_TOKENS, DIM_EMB),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(ETIQUETAS), activation="softmax"),
    ])
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Entrenando con {len(textos)} frases, {len(ETIQUETAS)} intenciones…")
    modelo.fit(X, y, epochs=260, verbose=2)

    # Guardar modelo (enteros) + vocabulario + etiquetas.
    modelo.save(RUTA_MODELO)
    vocab = vectorizer.get_vocabulary(include_special_tokens=False)
    with open(RUTA_VOCAB, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)
    with open(RUTA_ETIQUETAS, "w", encoding="utf-8") as fh:
        json.dump(ETIQUETAS, fh, ensure_ascii=False)

    # Prueba rápida (frases NO exactas del dataset).
    pruebas = [
        "necesito conectar la luz de mi casa nueva",
        "como va lo de mi solicitud",
        "que papeles tengo que presentar",
        "cuentame un chiste por favor",
        "hola buenas",
        "en que cosas me ayudas",
        "como empiezo un tramite nuevo",
        "como apruebo o rechazo un tramite",
        "como diseno el flujo de una politica",
        "como creo un usuario funcionario",
        "donde veo mi bandeja de entrada",
        "como invito a alguien a editar el diagrama",
    ]
    xp = vectorizer(tf.constant(pruebas))
    probs = modelo.predict(xp, verbose=0)
    print("\n=== prueba ===")
    for texto, p in zip(pruebas, probs):
        i = int(np.argmax(p))
        print(f"  '{texto}' -> {ETIQUETAS[i]} ({p[i]:.2f})")
    print(f"\nGuardado: {RUTA_MODELO}")


if __name__ == "__main__":
    entrenar()
