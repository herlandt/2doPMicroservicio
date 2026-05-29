"""CU-39 — voz a formulario."""
import json
import re

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.nlp import CampoSchema, CampoSugerido, VozAFormularioResponse

router = APIRouter(prefix="/nlp", tags=["nlp"])


@router.post("/voz-a-formulario", response_model=VozAFormularioResponse)
async def voz_a_formulario(
    audio: UploadFile = File(..., description="audio del funcionario"),
    schema_campos: str = Form(..., description="JSON con la lista de CampoSchema"),
) -> VozAFormularioResponse:
    """Transcribe el audio y mapea entidades al schema del formulario activo.

    **STUB**: por ahora devuelve un texto fijo y rellena los campos del schema
    con valores plausibles. Cuando se conecte Whisper + spaCy NER, este
    endpoint usará STT real + un parser por tipo de campo.
    """
    # Consumimos el audio para que el cliente no se quede esperando — en stub
    # no hacemos nada con él más allá de medir el tamaño.
    raw = await audio.read()
    _ = len(raw)

    try:
        campos: list[CampoSchema] = [CampoSchema(**c) for c in json.loads(schema_campos)]
    except (json.JSONDecodeError, TypeError, ValueError):
        campos = []

    texto = (
        "El nombre del cliente es Juan Perez, su DNI es 12345678, "
        "la inspeccion esta programada para mañana a las 10 de la mañana."
    )

    sugerencias = [_sugerir(c, texto) for c in campos]
    return VozAFormularioResponse(texto_transcrito=texto, campos=sugerencias)


def _sugerir(campo: CampoSchema, texto: str) -> CampoSugerido:
    """Mapeo muy básico: detecta patrones simples por tipo. Solo para el stub."""
    nombre = campo.nombre.lower()
    if "dni" in nombre or "ci" in nombre:
        m = re.search(r"\b\d{6,9}\b", texto)
        return CampoSugerido(campo=campo.nombre, valor=m.group(0) if m else "", confianza=0.95 if m else 0.2)
    if "nombre" in nombre:
        return CampoSugerido(campo=campo.nombre, valor="Juan Perez", confianza=0.92)
    if "fecha" in nombre:
        return CampoSugerido(campo=campo.nombre, valor="", confianza=0.60)
    return CampoSugerido(campo=campo.nombre, valor="", confianza=0.30)
