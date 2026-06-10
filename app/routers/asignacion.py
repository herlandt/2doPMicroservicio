"""CU-40 — asignación automática de política.

Usa un modelo TensorFlow PROPIO y REENTRENABLE (app/ml/clasificador_politica).
Si el modelo aún no está entrenado o no conoce ninguna de las políticas activas,
cae a una heurística de solapamiento de palabras para no romper nunca el flujo.
El backend reentrena el modelo (POST /asignacion/reentrenar) al crear/activar
políticas, tal como exige el enunciado P2 §3.2.2.
"""
import logging
import re
import unicodedata

from fastapi import APIRouter

from app.ml import clasificador_politica
from app.schemas.asignacion import (
    AsignacionRequest,
    AsignacionResponse,
    Candidato,
    ReentrenarRequest,
    ReentrenarResponse,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/asignacion", tags=["asignacion"])

_STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "y", "o", "a", "en", "para",
    "por", "con", "del", "al", "que", "se", "su", "mi", "tu", "lo", "le", "es",
    "como", "necesito", "quiero", "tramite", "hacer", "proceso", "solicitud",
}


@router.post("/politica", response_model=AsignacionResponse)
def asignar_politica(req: AsignacionRequest) -> AsignacionResponse:
    """Clasifica la descripción del cliente en una de las políticas activas."""
    if not req.politicas_activas:
        return AsignacionResponse(politica_id="", confianza=0.0, top3=[], fuente="heuristica")

    descripcion = (req.descripcion or "").strip()
    if not descripcion:
        return AsignacionResponse(politica_id="", confianza=0.0, top3=[], fuente="heuristica")

    # 1) Modelo TensorFlow (si está entrenado y conoce alguna política activa).
    try:
        pred = clasificador_politica.clasificar_sobre(descripcion, req.politicas_activas)
    except Exception as e:  # noqa: BLE001 — degradar a heurística
        log.warning("Clasificador de política falló, uso heurística: %s", e)
        pred = None
    if pred is not None:
        pid, conf, top3 = pred
        return AsignacionResponse(
            politica_id=pid,
            confianza=conf,
            top3=[Candidato(**c) for c in top3],
            fuente="modelo",
        )

    # 2) Respaldo heurístico: solapamiento de palabras de la descripción con el
    #    texto (nombre + descripción + categoría + palabras_clave) de cada política.
    desc_tokens = _tokens(descripcion)
    scored: list[tuple[float, Candidato]] = []
    for p in req.politicas_activas:
        texto_pol = " ".join([p.nombre, p.descripcion, p.categoria, " ".join(p.palabras_clave)])
        score = _solapamiento(desc_tokens, _tokens(texto_pol), p.nombre, descripcion)
        scored.append((score, Candidato(politica_id=p.id, nombre=p.nombre, confianza=score)))

    scored.sort(key=lambda x: x[0], reverse=True)
    total = sum(s for s, _ in scored) or 1.0
    normalizado = [
        Candidato(politica_id=c.politica_id, nombre=c.nombre, confianza=round(s / total, 3))
        for s, c in scored
    ]
    return AsignacionResponse(
        politica_id=normalizado[0].politica_id,
        confianza=normalizado[0].confianza,
        top3=normalizado[:3],
        fuente="heuristica",
    )


@router.post("/reentrenar", response_model=ReentrenarResponse)
def reentrenar(req: ReentrenarRequest) -> ReentrenarResponse:
    """Reentrena el clasificador de política con las políticas dadas (CU-40,
    P2 §3.2.2). Lo llama el backend al crear/activar una política."""
    try:
        stats = clasificador_politica.entrenar([p.model_dump() for p in req.politicas])
    except Exception as e:  # noqa: BLE001
        log.warning("Reentrenamiento de política falló: %s", e)
        return ReentrenarResponse(entrenado=False, motivo=f"{type(e).__name__}: {e}")
    return ReentrenarResponse(**stats)


# ── heurística de respaldo ───────────────────────────────────────────────────
def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.lower()


def _tokens(texto: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", _norm(texto)) if len(w) > 2 and w not in _STOPWORDS}


def _solapamiento(desc: set[str], pol: set[str], nombre: str, descripcion: str) -> float:
    """Cuenta palabras compartidas + bonus si el nombre de la política aparece."""
    score = float(len(desc & pol))
    if nombre and _norm(nombre) in _norm(descripcion):
        score += 2.0
    return score if score > 0 else 0.01
