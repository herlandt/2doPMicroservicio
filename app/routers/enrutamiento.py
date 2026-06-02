"""CU-42 / CU-43 / CU-44 / CU-45 — motor de enrutamiento (stubs)."""
import random
from collections import Counter

from fastapi import APIRouter

from app.schemas.enrutamiento import (
    Anomalia,
    AnomaliasRequest,
    PasoOmitido,
    PrioridadesRequest,
    RiesgoDemoraRequest,
    RutaOptimaRequest,
    RutaOptimaResponse,
    TramitePriorizado,
    TramiteRiesgo,
)

router = APIRouter(prefix="/enrutamiento", tags=["enrutamiento"])


# ── CU-42 · Ruta óptima ──────────────────────────────────────────────────────

@router.post("/ruta-optima", response_model=RutaOptimaResponse)
def ruta_optima(req: RutaOptimaRequest) -> RutaOptimaResponse:
    """Devuelve la ruta sugerida.

    **STUB**: devuelve los nodos en el orden recibido, sin omitir pasos.
    """
    nodos_ids = [n.get("id", "") for n in req.nodos_politica if n.get("id")]
    return RutaOptimaResponse(
        ruta_sugerida=nodos_ids,
        pasos_omitidos=[],
        confianza=0.6,
        explicacion="STUB: el motor IA aún no está entrenado; se sigue la ruta completa.",
    )


# ── CU-43 · Riesgo de demora ─────────────────────────────────────────────────

@router.post("/riesgo-demora", response_model=list[TramiteRiesgo])
def riesgo_demora(req: RiesgoDemoraRequest) -> list[TramiteRiesgo]:
    """**STUB** que combina las features de forma simple para dar un nivel."""
    salida: list[TramiteRiesgo] = []
    for t in req.tramites:
        # Ausencia real de datos: sin carga ni complejidad no hay base para estimar.
        if t.carga_departamento == 0 and t.complejidad == 0:
            nivel = "desconocido"
            score = 0.0
            razones = ["sin datos suficientes"]
        else:
            # Heurística determinista para el stub.
            score = min(1.0, 0.2 + 0.4 * t.carga_departamento + 0.4 * t.complejidad)
            if score >= 0.8:
                nivel = "alto"
                razones = ["carga alta del departamento", "trámite complejo"]
            elif score >= 0.5:
                nivel = "medio"
                razones = ["carga moderada"]
            else:
                nivel = "bajo"
                razones = ["dentro de parámetros normales"]
        salida.append(TramiteRiesgo(
            tramite_id=t.tramite_id,
            prob_superar_sla=round(score, 3),
            nivel=nivel,
            razones=razones,
        ))
    return salida


# ── CU-44 · Prioridades ──────────────────────────────────────────────────────

@router.post("/prioridades", response_model=list[TramitePriorizado])
def prioridades(req: PrioridadesRequest) -> list[TramitePriorizado]:
    """**STUB**: ordena por riesgo declarado y prioridad manual."""
    nivel_score = {"alto": 1.0, "medio": 0.7, "bajo": 0.4, None: 0.5, "desconocido": 0.5}
    salida: list[TramitePriorizado] = []
    for t in req.tramites:
        base = nivel_score.get(t.riesgo_demora, 0.5)
        # I1: prioridad manual mayor = más urgente (backend: 1=baja, 2=media, 3=alta).
        # NO invertir: un trámite urgente debe puntuar más, no menos.
        prio = (min(max(t.prioridad_manual, 1), 3) - 1) / 2.0
        score = round(0.7 * base + 0.3 * prio, 3)
        salida.append(TramitePriorizado(
            tramite_id=t.tramite_id,
            score=score,
            motivo=f"riesgo={t.riesgo_demora or 'desconocido'}, prioridad={t.prioridad_manual}",
        ))
    salida.sort(key=lambda x: x.score, reverse=True)
    return salida


# ── CU-45 · Anomalías ────────────────────────────────────────────────────────

@router.post("/anomalias", response_model=list[Anomalia])
def anomalias(req: AnomaliasRequest) -> list[Anomalia]:
    """**STUB**: marca como anomalía secuencias largas o con loops obvios."""
    salida: list[Anomalia] = []
    for sec in req.secuencias:
        transiciones = sec.transiciones
        if not transiciones:
            continue

        # Loop: el mismo nodo aparece más de 2 veces
        nodos = [t.get("nodo", "") for t in transiciones]
        conteo = Counter(nodos)
        loops = {n for n, c in conteo.items() if c > 2}
        if loops:
            salida.append(Anomalia(
                tramite_id=sec.tramite_id,
                categoria="loop_derivaciones",
                score=0.9,
                descripcion=f"El nodo {next(iter(loops))} aparece más de 2 veces",
            ))
            continue

        # Tiempo atípico: alguna transición > 24h
        if any((t.get("delta_segundos", 0) or 0) > 86_400 for t in transiciones):
            salida.append(Anomalia(
                tramite_id=sec.tramite_id,
                categoria="tiempo_atipico",
                score=0.7,
                descripcion="Alguna transición superó las 24 horas",
            ))
    return salida


# ── Reentrenamiento ──────────────────────────────────────────────────────────

@router.post("/modelos/reentrenar")
def reentrenar(modelo: str | None = None) -> dict:
    """**STUB** — devuelve un job-id ficticio. Cuando haya modelos reales, este
    endpoint dispara el script de entrenamiento en background."""
    return {
        "jobId": f"job-{random.randint(1000, 9999)}",
        "estado": "stub",
        "modelo": modelo or "todos",
    }
