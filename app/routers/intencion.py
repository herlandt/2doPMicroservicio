"""CU-46 — Clasificación de intención del asistente con un modelo TensorFlow
propio (entrenado en app/ml). El backend Spring lo usa para entender qué quiere
el usuario y responder con datos reales."""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

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
