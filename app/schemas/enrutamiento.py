"""CU-42 / CU-43 / CU-44 / CU-45 — motor de enrutamiento."""
from pydantic import BaseModel
from typing import Any


# ── CU-42 · Ruta óptima ──────────────────────────────────────────────────────

class RutaOptimaRequest(BaseModel):
    tramite_id: str
    politica_id: str
    contexto: dict[str, Any] = {}   # tipo cliente, hora, datos del expediente, etc.
    nodos_politica: list[dict[str, Any]] = []   # nodos del flujo de la política


class PasoOmitido(BaseModel):
    nodo_id: str
    motivo: str


class RutaOptimaResponse(BaseModel):
    ruta_sugerida: list[str]
    pasos_omitidos: list[PasoOmitido] = []
    confianza: float
    explicacion: str | None = None


# ── CU-43 · Riesgo de demora ─────────────────────────────────────────────────

class TramiteFeatures(BaseModel):
    tramite_id: str
    carga_departamento: float = 0
    complejidad: float = 0
    hora_dia: int = 0
    dia_semana: int = 0


class RiesgoDemoraRequest(BaseModel):
    tramites: list[TramiteFeatures]


class TramiteRiesgo(BaseModel):
    tramite_id: str
    prob_superar_sla: float
    nivel: str   # bajo | medio | alto | desconocido
    razones: list[str] = []


# ── CU-44 · Prioridades ──────────────────────────────────────────────────────

class TramitePendiente(BaseModel):
    tramite_id: str
    politica_id: str | None = None
    fecha_inicio: str | None = None
    prioridad_manual: int = 3
    riesgo_demora: str | None = None


class PrioridadesRequest(BaseModel):
    funcionario_id: str
    tramites: list[TramitePendiente]


class TramitePriorizado(BaseModel):
    tramite_id: str
    score: float
    motivo: str


# ── CU-45 · Anomalías ────────────────────────────────────────────────────────

class SecuenciaTramite(BaseModel):
    tramite_id: str
    transiciones: list[dict[str, Any]] = []   # [{nodo_anterior, nodo, delta_segundos}]


class AnomaliasRequest(BaseModel):
    secuencias: list[SecuenciaTramite]


class Anomalia(BaseModel):
    tramite_id: str
    categoria: str   # tiempo_atipico | secuencia_inusual | loop_derivaciones | salto_no_autorizado
    score: float
    descripcion: str
