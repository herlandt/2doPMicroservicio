"""CU-41 — reportes ad-hoc por consulta natural."""
import re
import unicodedata
from datetime import date, datetime, timezone

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
    # Texto sin acentos para comparaciones robustas ("trámite" -> "tramite")
    consulta_norm = _quitar_acentos(consulta)

    # Plantilla 1 — trámites en un rango de fechas
    rango = _detectar_rango_fechas(consulta_norm)
    if "tramite" in consulta_norm and rango:
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
    if "cuantos" in consulta_norm or "conteo" in consulta_norm or "agrupar" in consulta_norm:
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


def _quitar_acentos(texto: str) -> str:
    """Devuelve el texto sin marcas diacríticas (acentos) para comparar.

    Usa NFKD para descomponer y filtra los caracteres combinantes (Mn)."""
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


_MESES_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def _detectar_rango_fechas(texto: str) -> tuple[str, str] | None:
    """Detecta 'entre el X y el Y de mes' devolviendo ISO date strings.

    Si el texto menciona un mes en español (y opcionalmente un año de 4 cifras)
    se usa ese mes/año; si no se menciona mes/año explícito, se asume el mes en
    curso (año/mes actual en UTC). Stub muy ingenuo — sirve solo para demos."""
    # Buscar dos números cercanos a "entre"
    m = re.search(r"entre\s+(?:el\s+)?(\d{1,2})\s+y\s+(?:el\s+)?(\d{1,2})", texto)
    if not m:
        return None
    d1, d2 = int(m.group(1)), int(m.group(2))
    # Validar días en 1..31 y normalizar orden
    if not (1 <= d1 <= 31 and 1 <= d2 <= 31):
        return None
    d1, d2 = sorted((d1, d2))

    # Mes/año por defecto: el mes en curso (UTC)
    hoy = datetime.now(timezone.utc)
    anio, mes = hoy.year, hoy.month

    # Detectar mes en español si aparece en el texto
    mm = re.search(r"\b(" + "|".join(_MESES_ES) + r")\b", texto)
    if mm:
        mes = _MESES_ES[mm.group(1)]
    # Detectar año de 4 cifras si aparece, validando que sea un año plausible
    ma = re.search(r"\b(\d{4})\b", texto)
    if ma:
        cand = int(ma.group(1))
        anio = cand if 2000 <= cand <= 2100 else anio

    # Validar que los días existan para ese mes/año (rechaza p.ej. 31 de febrero)
    try:
        date(anio, mes, d1)
        date(anio, mes, d2)
    except ValueError:
        return None

    desde = f"{anio:04d}-{mes:02d}-{d1:02d}T00:00:00Z"
    hasta = f"{anio:04d}-{mes:02d}-{d2:02d}T23:59:59Z"
    return desde, hasta
