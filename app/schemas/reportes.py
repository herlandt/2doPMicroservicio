"""CU-41 — reportes ad-hoc por consulta natural."""
from pydantic import BaseModel
from typing import Any


class ReporteNaturalRequest(BaseModel):
    consulta: str


class ColumnaReporte(BaseModel):
    nombre: str
    tipo: str  # string|number|date|boolean


class ReporteNaturalResponse(BaseModel):
    """El microservicio interpreta y devuelve la consulta como pipeline MongoDB.
    Spring lo valida (whitelist de colecciones, sin $out/$merge/$function/$accumulator)
    y lo ejecuta.
    """
    collection: str
    pipeline: list[dict[str, Any]]
    columnas: list[ColumnaReporte]
