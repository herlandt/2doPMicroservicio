"""Liveness y readiness para que Spring sepa cuándo el microservicio está listo."""
from fastapi import APIRouter

from app.ml import clasificador, clasificador_politica, enrutamiento_modelos
from app.schemas.common import HealthResponse, ReadyResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
def healthz() -> HealthResponse:
    """Liveness probe — el servicio está vivo."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadyResponse, tags=["health"])
def readyz() -> ReadyResponse:
    """Readiness probe — estado real de cada modelo.

    'ready' = modelo TensorFlow entrenado en disco; 'heuristico' = funciona con
    reglas/heurística (aún sin modelo TF); 'stub' = pendiente.
    """
    enr = "ready" if enrutamiento_modelos.disponible() else "stub"
    return ReadyResponse(
        ready=True,
        models={
            "clasificador_intencion": "ready" if clasificador.disponible() else "stub",
            "clasificador_politica": "ready" if clasificador_politica.disponible() else "heuristico",
            "voz_a_formulario": "heuristico",
            "ruta_optima": enr,
            "riesgo_sla": enr,
            "prioridades": enr,
            "anomalias": enr,
            "nlp_to_pipeline": "stub",
        },
    )
