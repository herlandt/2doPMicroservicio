"""Liveness y readiness para que Spring sepa cuándo el microservicio está listo."""
from fastapi import APIRouter

from app.schemas.common import HealthResponse, ReadyResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
def healthz() -> HealthResponse:
    """Liveness probe — el servicio está vivo."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadyResponse, tags=["health"])
def readyz() -> ReadyResponse:
    """Readiness probe — qué modelos están cargados.

    Por ahora todos son stubs, así que reporta 'stub' en cada uno. Cuando se
    carguen modelos TensorFlow reales, este endpoint dirá 'ready'.
    """
    return ReadyResponse(
        ready=True,
        models={
            "clasificador_politica": "stub",
            "voz_a_formulario": "stub",
            "ruta_optima": "stub",
            "riesgo_sla": "stub",
            "prioridades": "stub",
            "anomalias": "stub",
            "nlp_to_pipeline": "stub",
        },
    )
