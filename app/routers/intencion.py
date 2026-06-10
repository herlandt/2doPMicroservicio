"""CU-46 — Clasificación de intención del asistente con un modelo TensorFlow
propio (entrenado en app/ml). El backend Spring lo usa para entender qué quiere
el usuario y responder con datos reales."""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app import aws_ai, gemini
from app.config import settings
from app.ml import clasificador

router = APIRouter(prefix="/nlp", tags=["nlp"])


class IntencionRequest(BaseModel):
    consulta: str


class IntencionResponse(BaseModel):
    intencion: str
    confianza: float


@router.post("/clasificar-intencion", response_model=IntencionResponse)
def clasificar_intencion(req: IntencionRequest) -> IntencionResponse:
    if not clasificador.disponible():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de intención no entrenado. Corre 'python -m app.ml.entrenar'.",
        )
    try:
        intencion, confianza = clasificador.clasificar(req.consulta)
    except Exception as e:  # TF no instalado o modelo ilegible -> degradar
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Clasificador no disponible: {type(e).__name__}",
        )
    return IntencionResponse(intencion=intencion, confianza=round(confianza, 3))


class AsistenteRequest(BaseModel):
    consulta: str
    contexto: str | None = None


class AsistenteResponse(BaseModel):
    respuesta: str


@router.post("/asistente", response_model=AsistenteResponse)
def asistente(req: AsistenteRequest) -> AsistenteResponse:
    """CU-31 (híbrido): respuesta generativa con Bedrock para los casos que el
    clasificador TensorFlow NO resuelve (baja confianza / fuera de alcance).
    Devuelve 503 si AWS está apagado o Bedrock falla → el backend Spring degrada
    a su base de conocimiento local. Así Bedrock solo se invoca en la minoría de
    consultas difíciles y el gasto se mantiene mínimo."""
    prov = (settings.ia_provider or "local").lower()
    if prov not in ("gemini", "aws"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Asistente LLM no habilitado (IA_PROVIDER=local).",
        )
    try:
        if prov == "gemini":
            texto = gemini.responder_asistente(req.consulta, req.contexto or "")
        else:
            texto = aws_ai.responder_asistente(req.consulta, req.contexto or "")
    except Exception as e:  # noqa: BLE001 — degradar en el backend (KB local)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Asistente LLM no disponible: {type(e).__name__}",
        )
    return AsistenteResponse(respuesta=texto)
