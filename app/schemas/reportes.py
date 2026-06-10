"""CU-41 — reportes ad-hoc por consulta natural."""
from pydantic import BaseModel
from typing import Any


class ReporteNaturalRequest(BaseModel):
    consulta: str


class ColumnaReporte(BaseModel):
    nombre: str
    tipo: str  # string|number|date|boolean


class ReporteNaturalResponse(BaseModel):
    """Plan de reporte interpretado por el microservicio (NLP).

    El micro interpreta la consulta y devuelve:
      - ``collection`` + ``pipeline``: consulta MongoDB SEGURA (sin $lookup/$out…)
        que el backend valida y ejecuta.
      - ``enriquecer``: qué nombres resolver en el backend vía repositorios
        (``cliente_nombre`` | ``politica_nombre`` | ``departamento_nombre``) — así
        evitamos joins frágiles en Mongo y reusamos los repos de Spring.
      - ``filtros_post``: filtros por NOMBRE que el backend resuelve a ids
        (p. ej. ``politica_nombre`` → ids) o aplica tras enriquecer.
      - ``operacion`` / ``agrupar_por``: pistas para la UI (tabla vs gráfico).
    """
    collection: str
    pipeline: list[dict[str, Any]]
    columnas: list[ColumnaReporte] = []
    enriquecer: list[str] = []
    filtros_post: dict[str, str] = {}
    operacion: str = "listar"          # listar | contar
    agrupar_por: str | None = None     # estado | politica | departamento
