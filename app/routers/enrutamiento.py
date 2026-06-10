"""CU-42 / CU-43 / CU-44 / CU-45 — motor de enrutamiento.

Usa modelos TensorFlow PROPIOS (app/ml/enrutamiento_modelos): riesgo, prioridad,
anomalía (autoencoder) y ruta. Son dominio-agnósticos (solo señales operativas).
Si algún modelo no está entrenado, cae a una heurística determinista equivalente
para no romper el flujo. El endpoint /modelos/reentrenar reentrena de verdad.
"""
import logging
from collections import Counter
from datetime import datetime

from fastapi import APIRouter

from app.ml import enrutamiento_modelos as em
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

log = logging.getLogger(__name__)
router = APIRouter(prefix="/enrutamiento", tags=["enrutamiento"])

_NIVEL_A_SCORE = {"alto": 0.9, "medio": 0.6, "bajo": 0.3, "desconocido": 0.5, None: 0.5}


# ── CU-42 · Ruta óptima ──────────────────────────────────────────────────────
@router.post("/ruta-optima", response_model=RutaOptimaResponse)
def ruta_optima(req: RutaOptimaRequest) -> RutaOptimaResponse:
    """Sugiere la ruta; omite nodos OPCIONALES cuya necesidad (modelo TF) es baja."""
    nodos = [n for n in req.nodos_politica if n.get("id")]
    nodos.sort(key=lambda n: n.get("orden", 0))
    total = len(nodos)

    ruta: list[str] = []
    omitidos: list[PasoOmitido] = []
    necesidades: list[float] = []
    usado_modelo = em.disponible()

    for n in nodos:
        nec = em.predecir_necesidad_nodo(
            str(n.get("tipo", "")), bool(n.get("opcional", False)),
            int(n.get("orden", 0) or 0), total)
        if nec is None:  # sin modelo → ruta completa (heurística)
            ruta.append(n["id"])
            continue
        necesidades.append(nec)
        if n.get("opcional") and nec < 0.5:
            omitidos.append(PasoOmitido(
                nodo_id=n["id"],
                motivo=f"Paso opcional con baja necesidad estimada ({nec:.2f})"))
        else:
            ruta.append(n["id"])

    if not usado_modelo:
        return RutaOptimaResponse(
            ruta_sugerida=[n["id"] for n in nodos], pasos_omitidos=[], confianza=0.6,
            explicacion="Modelo de ruta no entrenado; se sigue la ruta completa.")

    conf = round(sum(necesidades) / len(necesidades), 3) if necesidades else 0.7
    expl = (f"Se omiten {len(omitidos)} paso(s) opcional(es) de baja necesidad."
            if omitidos else "Todos los pasos son necesarios para este trámite.")
    return RutaOptimaResponse(ruta_sugerida=ruta, pasos_omitidos=omitidos,
                              confianza=conf, explicacion=expl)


# ── CU-43 · Riesgo de demora ─────────────────────────────────────────────────
@router.post("/riesgo-demora", response_model=list[TramiteRiesgo])
def riesgo_demora(req: RiesgoDemoraRequest) -> list[TramiteRiesgo]:
    salida: list[TramiteRiesgo] = []
    for t in req.tramites:
        # Sin datos no se estima (igual que antes).
        if t.carga_departamento == 0 and t.complejidad == 0:
            salida.append(TramiteRiesgo(tramite_id=t.tramite_id, prob_superar_sla=0.0,
                                        nivel="desconocido", razones=["sin datos suficientes"]))
            continue
        prob = em.predecir_riesgo(t.carga_departamento, t.complejidad, t.hora_dia, t.dia_semana)
        if prob is None:  # fallback heurístico
            prob = min(1.0, 0.2 + 0.4 * t.carga_departamento + 0.4 * t.complejidad)
        nivel = "alto" if prob >= 0.66 else "medio" if prob >= 0.4 else "bajo"
        salida.append(TramiteRiesgo(
            tramite_id=t.tramite_id, prob_superar_sla=round(prob, 3),
            nivel=nivel, razones=_razones_riesgo(t)))
    return salida


def _razones_riesgo(t) -> list[str]:
    r = []
    if t.carga_departamento >= 0.6:
        r.append("carga alta del departamento")
    if t.complejidad >= 0.6:
        r.append("trámite complejo")
    if t.hora_dia >= 15:
        r.append("ingresó en horario pico")
    if t.dia_semana in (6, 7):
        r.append("ingresó en fin de semana")
    return r or ["dentro de parámetros normales"]


# ── CU-44 · Prioridades ──────────────────────────────────────────────────────
@router.post("/prioridades", response_model=list[TramitePriorizado])
def prioridades(req: PrioridadesRequest) -> list[TramitePriorizado]:
    salida: list[TramitePriorizado] = []
    for t in req.tramites:
        riesgo_score = _NIVEL_A_SCORE.get(t.riesgo_demora, 0.5)
        espera = _dias_espera(t.fecha_inicio)
        score = em.predecir_prioridad(riesgo_score, t.prioridad_manual, espera)
        if score is None:  # fallback heurístico
            prio = (min(max(t.prioridad_manual, 1), 3) - 1) / 2.0
            score = 0.7 * riesgo_score + 0.3 * prio
        salida.append(TramitePriorizado(
            tramite_id=t.tramite_id, score=round(score, 3),
            motivo=f"riesgo={t.riesgo_demora or 'desconocido'}, prioridad={t.prioridad_manual}, "
                   f"espera={espera:.1f}d"))
    salida.sort(key=lambda x: x.score, reverse=True)
    return salida


def _dias_espera(fecha_inicio: str | None) -> float:
    if not fecha_inicio:
        return 0.0
    try:
        f = datetime.fromisoformat(fecha_inicio.replace("Z", "+00:00"))
        # Comparar consistente: aware con aware, naive con naive.
        ahora = datetime.now(f.tzinfo) if f.tzinfo else datetime.now()
        return max(0.0, (ahora - f).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


# ── CU-45 · Anomalías ────────────────────────────────────────────────────────
@router.post("/anomalias", response_model=list[Anomalia])
def anomalias(req: AnomaliasRequest) -> list[Anomalia]:
    salida: list[Anomalia] = []
    for sec in req.secuencias:
        transiciones = sec.transiciones
        if not transiciones:
            continue
        res = em.score_anomalia(transiciones)
        if res is None:  # fallback de reglas
            anomalia = _anomalia_reglas(sec.tramite_id, transiciones)
            if anomalia:
                salida.append(anomalia)
            continue
        err, umbral = res
        if err <= umbral:
            continue
        feat = em.feat_secuencia(transiciones)
        categoria, desc = _categorizar_anomalia(feat)
        score = min(1.0, 0.5 + 0.5 * (err - umbral) / max(umbral, 1e-6))
        salida.append(Anomalia(tramite_id=sec.tramite_id, categoria=categoria,
                               score=round(score, 3), descripcion=desc))
    return salida


def _categorizar_anomalia(feat: list[float]) -> tuple[str, str]:
    # feat = [largo, loop_ratio, rep_max, max_delta, mean_delta]
    if feat[1] > 0.2 or feat[2] > 0.3:
        return "loop_derivaciones", "Secuencia con derivaciones repetidas (posible loop)"
    if feat[3] > 0.3:
        return "tiempo_atipico", "Alguna transición tardó un tiempo atípicamente alto"
    return "secuencia_inusual", "El recorrido del trámite se desvía del patrón normal"


def _anomalia_reglas(tramite_id: str, transiciones: list[dict]) -> Anomalia | None:
    nodos = [t.get("nodo", "") for t in transiciones]
    loops = {n for n, c in Counter(nodos).items() if c > 2}
    if loops:
        return Anomalia(tramite_id=tramite_id, categoria="loop_derivaciones", score=0.9,
                        descripcion=f"El nodo {next(iter(loops))} aparece más de 2 veces")
    if any((t.get("delta_segundos", 0) or 0) > 86_400 for t in transiciones):
        return Anomalia(tramite_id=tramite_id, categoria="tiempo_atipico", score=0.7,
                        descripcion="Alguna transición superó las 24 horas")
    return None


# ── Reentrenamiento ──────────────────────────────────────────────────────────
@router.post("/modelos/reentrenar")
def reentrenar(modelo: str | None = None) -> dict:
    """Reentrena los modelos de enrutamiento (CU-42/43/44/45). Hoy con datos
    sintéticos; cuando se acumulen métricas reales se entrenan sobre ellas."""
    try:
        stats = em.entrenar_todos()
        return {"estado": "ok", "modelo": modelo or "enrutamiento", "stats": stats}
    except Exception as e:  # noqa: BLE001
        log.warning("Reentrenamiento de enrutamiento falló: %s", e)
        return {"estado": "error", "detalle": f"{type(e).__name__}: {e}"}
