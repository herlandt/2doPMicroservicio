"""CU-41 — reportes ad-hoc por consulta natural.

Parser NLP por intención + slots (determinista, sin API externa): interpreta la
consulta y arma un PLAN de reporte (pipeline Mongo seguro + qué enriquecer +
filtros por nombre). El backend ejecuta el pipeline, resuelve los nombres
(cliente/política/departamento) vía repositorios y exporta a Excel/PDF.

Es dominio-agnóstico respecto al NEGOCIO: opera sobre el esquema del propio
sistema (trámites/usuarios/políticas/departamentos), igual sea una eléctrica o
una clínica.
"""
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
    q = _quitar_acentos((req.consulta or "").lower())

    operacion = "contar" if any(k in q for k in (
        "cuantos", "cuantas", "conteo", "contar", "cantidad", "numero de",
        "total de", "agrupa", "agrupar")) else "listar"

    agrupar = None
    if "departamento" in q or " area" in q:
        agrupar = "departamento"
    elif "politica" in q or "tipo de tramite" in q:
        agrupar = "politica"
    elif "estado" in q:
        agrupar = "estado"
    if operacion == "contar" and agrupar is None:
        agrupar = "estado"

    # ── filtros ──
    match: dict = {}
    rango = _detectar_rango_fechas(q)
    if rango:
        match["fechaInicio"] = {"$gte": {"$date": rango[0]}, "$lte": {"$date": rango[1]}}
    estado = _detectar_estado(q)
    if estado:
        match["estadoActual"] = {"$regex": f"^{estado}$", "$options": "i"}

    enriquecer: list[str] = []
    filtros_post: dict[str, str] = {}

    pol = _nombre_tras(q, ("politica", "tipo de tramite"))
    if pol:
        filtros_post["politica_nombre"] = pol
        enriquecer.append("politica_nombre")
    dep = _nombre_tras(q, ("departamento", "area"))
    if dep:
        filtros_post["departamento_nombre"] = dep
        enriquecer.append("departamento_nombre")

    if any(k in q for k in ("cliente", "solicitante", "nombre del")):
        enriquecer.append("cliente_nombre")
    if agrupar == "politica":
        enriquecer.append("politica_nombre")
    if agrupar == "departamento":
        enriquecer.append("departamento_nombre")
    enriquecer = list(dict.fromkeys(enriquecer))   # únicos, orden estable

    # ── construir pipeline ──
    pipeline: list[dict] = []
    if match:
        pipeline.append({"$match": match})

    if operacion == "contar":
        campo = {"estado": "$estadoActual", "politica": "$politicaId"}.get(agrupar, "$estadoActual")
        etiqueta = "estado" if agrupar == "estado" else "politicaId" if agrupar == "politica" else "estado"
        pipeline += [
            {"$group": {"_id": campo, "total": {"$sum": 1}}},
            {"$sort": {"total": -1}},
            {"$project": {"_id": 0, etiqueta: "$_id", "total": 1}},
        ]
        columnas = [ColumnaReporte(nombre=etiqueta, tipo="string"),
                    ColumnaReporte(nombre="total", tipo="number")]
    else:
        pipeline += [
            {"$project": {
                "_id": 0, "codigo": 1, "estadoActual": 1, "clienteId": 1,
                "politicaId": 1, "funcionarioActualId": 1, "nodoActualId": 1, "fechaInicio": 1,
            }},
            {"$limit": 2000},
        ]
        columnas = [
            ColumnaReporte(nombre="codigo", tipo="string"),
            ColumnaReporte(nombre="estadoActual", tipo="string"),
            ColumnaReporte(nombre="fechaInicio", tipo="date"),
        ]

    return ReporteNaturalResponse(
        collection="tramites", pipeline=pipeline, columnas=columnas,
        enriquecer=enriquecer, filtros_post=filtros_post,
        operacion=operacion, agrupar_por=agrupar,
    )


# ── NLP helpers ──────────────────────────────────────────────────────────────
_ESTADOS = {
    "aprobad": "Aprobado", "rechazad": "Rechazado", "observ": "Observado",
    "cancelad": "Cancelado", "en curso": "En curso", "curso": "En curso",
    "proceso": "En curso", "activ": "En curso", "pendient": "En curso",
}


def _detectar_estado(texto: str) -> str | None:
    for clave, valor in _ESTADOS.items():
        if clave in texto:
            return valor
    return None


def _nombre_tras(texto: str, claves: tuple[str, ...]) -> str | None:
    """Extrae el nombre que sigue a una palabra clave ('politica de X' -> 'X').

    Devuelve hasta 4 palabras; corta en conectores. El backend hace match difuso
    (contiene), así que basta con una extracción aproximada."""
    conectores = {"entre", "con", "por", "agrupad", "agrupar", "cuantos", "donde",
                  "que", "del", "los", "las", "en", "el", "la", "y"}
    for clave in claves:
        m = re.search(clave + r"\s+(?:de\s+|del\s+|la\s+|el\s+)?([a-z0-9 ]+)", texto)
        if not m:
            continue
        palabras = []
        for w in m.group(1).split():
            if w in conectores:
                break
            palabras.append(w)
            if len(palabras) >= 4:
                break
        nombre = " ".join(palabras).strip()
        if nombre:
            return nombre
    return None


def _quitar_acentos(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


_MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _detectar_rango_fechas(texto: str) -> tuple[str, str] | None:
    """Detecta 'entre el X y el Y de mes [año]' → (desde_iso, hasta_iso)."""
    m = re.search(r"entre\s+(?:el\s+)?(\d{1,2})\s+y\s+(?:el\s+)?(\d{1,2})", texto)
    if not m:
        return None
    d1, d2 = int(m.group(1)), int(m.group(2))
    if not (1 <= d1 <= 31 and 1 <= d2 <= 31):
        return None
    d1, d2 = sorted((d1, d2))
    hoy = datetime.now(timezone.utc)
    anio, mes = hoy.year, hoy.month
    mm = re.search(r"\b(" + "|".join(_MESES_ES) + r")\b", texto)
    if mm:
        mes = _MESES_ES[mm.group(1)]
    ma = re.search(r"\b(\d{4})\b", texto)
    if ma:
        cand = int(ma.group(1))
        anio = cand if 2000 <= cand <= 2100 else anio
    try:
        date(anio, mes, d1)
        date(anio, mes, d2)
    except ValueError:
        return None
    return (f"{anio:04d}-{mes:02d}-{d1:02d}T00:00:00Z",
            f"{anio:04d}-{mes:02d}-{d2:02d}T23:59:59Z")
