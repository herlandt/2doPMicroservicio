"""CU-40 — asignación automática de política."""
from pydantic import BaseModel


class PoliticaActiva(BaseModel):
    id: str
    nombre: str
    palabras_clave: list[str] = []
    # Texto adicional para la heurística de respaldo (cuando el modelo TF no está
    # entrenado todavía). El modelo TF usa solo el id para restringir candidatos.
    descripcion: str = ""
    categoria: str = ""


class AsignacionRequest(BaseModel):
    descripcion: str
    politicas_activas: list[PoliticaActiva]


class Candidato(BaseModel):
    politica_id: str
    nombre: str
    confianza: float


class AsignacionResponse(BaseModel):
    politica_id: str
    confianza: float
    top3: list[Candidato]
    # "modelo" → predicho por el clasificador TensorFlow; "heuristica" → respaldo.
    fuente: str = "heuristica"


# ── Reentrenamiento del clasificador de política (CU-40, P2 §3.2.2) ──
class PoliticaEntrenamiento(BaseModel):
    id: str
    nombre: str = ""
    descripcion: str = ""
    categoria: str = ""


class ReentrenarRequest(BaseModel):
    politicas: list[PoliticaEntrenamiento]


class ReentrenarResponse(BaseModel):
    entrenado: bool
    politicas: int = 0
    frases: int = 0
    accuracy: float | None = None
    motivo: str | None = None
