"""CU-14 — generación de diagrama de flujo por prompt (IA real).

El admin describe el proceso en lenguaje natural y la IA (Gemini, vía el provider
configurable) devuelve la estructura del diagrama: nodos + transiciones. El
backend Spring la materializa (mapea departamentos, enlaza actividades). Si el
provider es 'local' (o falla), devuelve 503 y el backend usa su heurística.
"""
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app import gemini
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/nlp", tags=["nlp"])


class DiagramaRequest(BaseModel):
    prompt: str
    departamentos: list[str] = []


@router.post("/diagrama")
def generar_diagrama(req: DiagramaRequest) -> dict:
    """Interpreta el prompt y devuelve {nodos, transiciones}. 503 → heurística."""
    prov = (settings.ia_provider or "local").lower()
    if prov != "gemini":
        # AWS Bedrock también podría hacerlo, pero hoy solo está cableado Gemini.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generación de diagrama por IA no habilitada (IA_PROVIDER!=gemini).",
        )
    try:
        return gemini.generar_flujo(req.prompt, req.departamentos)
    except Exception as e:  # noqa: BLE001 — el backend degrada a su heurística
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"IA de diagrama no disponible: {type(e).__name__}",
        )
