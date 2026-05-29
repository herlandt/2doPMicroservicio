"""CU-40 — asignación automática de política."""
from pydantic import BaseModel


class PoliticaActiva(BaseModel):
    id: str
    nombre: str
    palabras_clave: list[str] = []


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
