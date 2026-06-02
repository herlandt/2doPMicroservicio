"""CU-40 — asignación automática de política."""
from fastapi import APIRouter

from app.schemas.asignacion import (
    AsignacionRequest,
    AsignacionResponse,
    Candidato,
)

router = APIRouter(prefix="/asignacion", tags=["asignacion"])


@router.post("/politica", response_model=AsignacionResponse)
def asignar_politica(req: AsignacionRequest) -> AsignacionResponse:
    """Clasifica la descripción del cliente en una de las políticas activas.

    **STUB**: hace matching simple por palabra clave. Cuando se conecte el
    modelo TF de clasificación (embedding + softmax), reemplazar el cuerpo.
    """
    if not req.politicas_activas:
        # Sin políticas activas el clasificador no puede hacer nada.
        return AsignacionResponse(
            politica_id="",
            confianza=0.0,
            top3=[],
        )

    descripcion = req.descripcion.lower().strip()
    # Sin descripción no hay nada que clasificar (evita fabricar confianza de la nada).
    # OJO: con descripción real SÍ devolvemos candidatos aunque no haya match de
    # keyword — las políticas del stub apenas tienen palabras_clave, así que cortar
    # por "sin hits" dejaría la sugerencia siempre vacía (rompe el flujo CU-40).
    if not descripcion:
        return AsignacionResponse(politica_id="", confianza=0.0, top3=[])

    scored: list[tuple[float, Candidato]] = []
    for p in req.politicas_activas:
        score = _score(descripcion, p.palabras_clave, p.nombre)
        scored.append((score, Candidato(politica_id=p.id, nombre=p.nombre, confianza=score)))

    # Ordenar descendente y normalizar a softmax suave para que sumen ≈ 1.
    scored.sort(key=lambda x: x[0], reverse=True)

    total = sum(s for s, _ in scored) or 1.0
    normalizado = [
        Candidato(
            politica_id=c.politica_id,
            nombre=c.nombre,
            confianza=round(s / total, 3),
        )
        for s, c in scored
    ]

    return AsignacionResponse(
        politica_id=normalizado[0].politica_id,
        confianza=normalizado[0].confianza,
        top3=normalizado[:3],
    )


def _score(descripcion: str, palabras_clave: list[str], nombre_politica: str) -> float:
    """Heurística simple: cuenta hits + bonus si el nombre aparece en el texto."""
    score = 0.0
    if not descripcion:
        return score
    for kw in palabras_clave:
        if kw and kw.lower() in descripcion:
            score += 1.0
    # Bonus por nombre
    if nombre_politica and nombre_politica.lower() in descripcion:
        score += 2.0
    # Si no hubo hits, dar un baseline muy pequeño para evitar dividir por cero.
    return score if score > 0 else 0.01
