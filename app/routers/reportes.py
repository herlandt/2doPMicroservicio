"""CU-41 — reportes ad-hoc por consulta natural."""
import re

from fastapi import APIRouter

from app.schemas.reportes import (
    ColumnaReporte,
    ReporteNaturalRequest,
    ReporteNaturalResponse,
)

router = APIRouter(prefix="/reportes", tags=["reportes"])


@router.post("/consulta-natural", response_model=ReporteNaturalResponse)
def consulta_natural(req: ReporteNaturalRequest) -> ReporteNaturalResponse:
    """Interpreta la consulta y devuelve un pipeline MongoDB para que Spring lo ejecute.

    **STUB**: tres plantillas básicas. Cuando se conecte un parser NLP real
    (spaCy + reglas o un LLM local), reemplazar.
    """
    consulta = req.consulta.lower()

    # Plantilla 1 — trámites en un rango de fechas
    rango = _detectar_rango_fechas(consulta)
    if "tramite" in consulta and rango:
        desde_iso, hasta_iso = rango
        return ReporteNaturalResponse(
            collection="tramites",
            pipeline=[
                {"$match": {
                    "fechaInicio": {
                        "$gte": {"$date": desde_iso},
                        "$lte": {"$date": hasta_iso},
                    }
                }},
                {"$project": {
                    "codigo": 1,
                    "estadoActual": 1,
                    "clienteId": 1,
                    "fechaInicio": 1,
                    "_id": 0,
                }},
                {"$limit": 50_000},
            ],
            columnas=[
                ColumnaReporte(nombre="codigo", tipo="string"),
                ColumnaReporte(nombre="estadoActual", tipo="string"),
                ColumnaReporte(nombre="clienteId", tipo="string"),
                ColumnaReporte(nombre="fechaInicio", tipo="date"),
            ],
        )

    # Plantilla 2 — conteo por estado
    if "cuantos" in consulta or "conteo" in consulta or "agrupar" in consulta:
        return ReporteNaturalResponse(
            collection="tramites",
            pipeline=[
                {"$group": {"_id": "$estadoActual", "total": {"$sum": 1}}},
                {"$sort": {"total": -1}},
                {"$project": {"estado": "$_id", "total": 1, "_id": 0}},
            ],
            columnas=[
                ColumnaReporte(nombre="estado", tipo="string"),
                ColumnaReporte(nombre="total", tipo="number"),
            ],
        )

    # Default — listado simple
    return ReporteNaturalResponse(
        collection="tramites",
        pipeline=[
            {"$project": {
                "codigo": 1,
                "estadoActual": 1,
                "fechaInicio": 1,
                "_id": 0,
            }},
            {"$limit": 1000},
        ],
        columnas=[
            ColumnaReporte(nombre="codigo", tipo="string"),
            ColumnaReporte(nombre="estadoActual", tipo="string"),
            ColumnaReporte(nombre="fechaInicio", tipo="date"),
        ],
    )


def _detectar_rango_fechas(texto: str) -> tuple[str, str] | None:
    """Detecta 'entre el X y el Y de mes' devolviendo ISO date strings.
    Stub muy ingenuo — sirve solo para demos."""
    # Buscar dos números cercanos a "entre"
    m = re.search(r"entre\s+(?:el\s+)?(\d{1,2})\s+y\s+(?:el\s+)?(\d{1,2})", texto)
    if not m:
        return None
    d1, d2 = int(m.group(1)), int(m.group(2))
    # Suponemos mayo 2025 si no detectamos mes — demo
    desde = f"2025-05-{d1:02d}T00:00:00Z"
    hasta = f"2025-05-{d2:02d}T23:59:59Z"
    return desde, hasta
