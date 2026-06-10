"""CU-39 — voz a formulario."""
import json
import logging
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app import aws_ai, gemini
from app.config import settings
from app.schemas.nlp import CampoSchema, CampoSugerido, VozAFormularioResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/nlp", tags=["nlp"])

# Tamaño de bloque para leer el audio en streaming (1 MB).
_CHUNK_SIZE = 1024 * 1024

# Texto de respaldo cuando NO hay STT real (AWS off) o cuando Transcribe falla:
# así el mapeo heurístico local sigue teniendo algo sobre lo que trabajar y la
# demo nunca se rompe.
_TEXTO_STUB = (
    "El nombre del cliente es Juan Perez, su DNI es 12345678, "
    "la inspeccion es el 15 de marzo de 2026 a las 10 de la mañana."
)


@router.post("/voz-a-formulario", response_model=VozAFormularioResponse)
async def voz_a_formulario(
    audio: UploadFile = File(..., description="audio del funcionario"),
    schema_campos: str = Form(..., description="JSON con la lista de CampoSchema"),
) -> VozAFormularioResponse:
    """Transcribe el audio y mapea entidades al schema del formulario activo.

    Dos modos (config ``AWS_AI_ENABLED``):
      • **AWS ON**  → Amazon Transcribe (STT real) + Amazon Bedrock (extracción
        estructurada contra el esquema del formulario).
      • **AWS OFF** → texto de respaldo + mapeo heurístico local (``_sugerir``).

    En modo AWS, si Transcribe o Bedrock fallan, se degrada al heurístico local
    para que la demo nunca se rompa.
    """
    # Leemos el audio completo respetando el límite de tamaño (413 si excede).
    buffer = bytearray()
    try:
        while chunk := await audio.read(_CHUNK_SIZE):
            buffer.extend(chunk)
            if len(buffer) > settings.max_audio_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"El audio excede el máximo permitido ({settings.max_audio_bytes} bytes).",
                )
    finally:
        await audio.close()

    try:
        campos: list[CampoSchema] = [CampoSchema(**c) for c in json.loads(schema_campos)]
    except (json.JSONDecodeError, TypeError, ValueError):
        campos = []

    # ── Dictado según el proveedor (IA_PROVIDER) ──
    #   gemini → 1 llamada multimodal: audio → (transcripción + campos).
    #   aws    → Amazon Transcribe (STT) + Bedrock (extracción).
    #   local / cualquier fallo → texto de respaldo + heurística (_sugerir).
    texto = _TEXTO_STUB
    sugerencias: list[CampoSugerido] | None = None
    prov = (settings.ia_provider or "local").lower()

    if prov == "gemini":
        try:
            texto, sugerencias = gemini.extraer_de_audio(
                bytes(buffer), audio.content_type, [c.model_dump() for c in campos])
        except Exception as e:  # noqa: BLE001 — degradar a local
            log.warning("Gemini (dictado) falló, uso heurística local: %s", e)
            sugerencias = None
    elif prov == "aws":
        try:
            texto = aws_ai.transcribir(bytes(buffer), audio.filename, audio.content_type)
        except Exception as e:  # noqa: BLE001
            log.warning("Transcribe falló, uso texto de respaldo: %s", e)
        try:
            sugerencias = aws_ai.extraer_campos([c.model_dump() for c in campos], texto)
        except Exception as e:  # noqa: BLE001
            log.warning("Bedrock (extracción) falló, uso heurística local: %s", e)
            sugerencias = None

    if sugerencias is None:
        sugerencias = [_sugerir(c, texto) for c in campos]

    # es_stub: si el texto sigue siendo el de respaldo, NO hubo STT real — los
    # consumidores que usan la transcripción como dato del usuario deben ignorarla.
    return VozAFormularioResponse(
        texto_transcrito=texto, campos=sugerencias, es_stub=(texto is _TEXTO_STUB))


# ── Extracción de entidades (heurística determinista del stub) ───────────────
# Dado un TEXTO, mapeamos cada entidad al campo que corresponde por su
# nombre/tipo. Es lo que verifica el test "cada dato cae en su campo".

_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_DNI = re.compile(r"\b\d{6,9}\b")
_RE_TEL = re.compile(r"\b[67]\d{7}\b")  # móvil Bolivia: 8 dígitos, empieza 6/7
_RE_FECHA_NUM = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
_RE_FECHA_TXT = re.compile(
    r"\b(\d{1,2})\s+de\s+(" + "|".join(_MESES) + r")(?:\s+de\s+(\d{4}))?",
    re.IGNORECASE,
)
_RE_NOMBRE_KEY = re.compile(
    r"(?:me llamo|se llama|mi nombre es|nombre(?:\s+del\s+cliente|\s+completo)?\s+es|"
    r"el\s+cliente\s+es|solicitante\s+es)\s+(.+)",
    re.IGNORECASE,
)
# Palabras que cortan la captura del nombre (no son parte del nombre propio).
_STOP_NOMBRE = {
    "su", "el", "la", "los", "las", "de", "del", "con", "y", "o", "tiene", "es",
    "cedula", "cédula", "dni", "ci", "carnet", "numero", "número", "correo",
    "celular", "telefono", "teléfono", "inspeccion", "inspección", "fecha",
    "esta", "está", "para", "cliente",
}


def _extraer_nombre(texto: str) -> str:
    m = _RE_NOMBRE_KEY.search(texto)
    if not m:
        return ""
    palabras: list[str] = []
    for w in m.group(1).split():
        limpio = re.sub(r"[^A-Za-zÁÉÍÓÚÑáéíóúñ]", "", w)
        if not limpio or limpio.lower() in _STOP_NOMBRE:
            break
        palabras.append(limpio)
        if len(palabras) >= 3:
            break
    return " ".join(p.capitalize() for p in palabras)


def _extraer_fecha(texto: str) -> str:
    m = _RE_FECHA_NUM.search(texto)
    if m:
        d, mes, anio = m.groups()
        anio = anio if len(anio) == 4 else "20" + anio
        return f"{int(d):02d}/{int(mes):02d}/{anio}"
    m = _RE_FECHA_TXT.search(texto)
    if m:
        d, mes_txt, anio = m.group(1), m.group(2), m.group(3)
        mes = _MESES[mes_txt.lower()]
        return f"{int(d):02d}/{mes:02d}/{anio}" if anio else f"{int(d):02d}/{mes:02d}"
    bajo = texto.lower()
    if "mañana" in bajo:
        return "mañana"
    if "hoy" in bajo:
        return "hoy"
    return ""


def _tokens(nombre: str) -> set[str]:
    """Divide el nombre del campo en palabras (snake_case, espacios, guiones).
    Matchear por TOKEN evita falsos positivos como "ci" dentro de "cita"."""
    return {t for t in re.split(r"[^a-záéíóúñ]+", nombre.lower()) if t}


def _sugerir(campo: CampoSchema, texto: str) -> CampoSugerido:
    """Mapea el campo del formulario a la entidad del texto dictado según su
    nombre/tipo. Determinista (apto para test). El audio→texto sigue siendo
    stub (falta Whisper); esto resuelve el "rellenar el campo correcto"."""
    # El nombre técnico Y la etiqueta humana aportan palabras para el matching.
    toks = _tokens(campo.nombre) | _tokens(campo.etiqueta or "")
    tipo = (campo.tipo or "").lower()
    bajo = texto.lower()

    def has(*kws: str) -> bool:
        return any(k in toks for k in kws)

    def out(valor: str, conf_ok: float, conf_no: float = 0.25) -> CampoSugerido:
        return CampoSugerido(
            campo=campo.nombre, valor=valor, confianza=conf_ok if valor else conf_no
        )

    # Select: el valor dictado debe coincidir con una de las opciones definidas.
    if tipo == "select" and campo.opciones:
        elegido = next((o for o in campo.opciones if o and o.lower() in bajo), "")
        return out(elegido, 0.85, 0.25)

    # Checkbox (sí/no): detecta afirmación o negación explícita en el texto.
    if tipo == "checkbox":
        afirm = any(k in bajo for k in (
            "conforme", "cumple", "aprobado", "completo", "completos",
            "de acuerdo", "correcto", "sí ", "afirmativo",
        ))
        nega = any(k in bajo for k in (
            "no conforme", "incorrecto", "rechaz", "falta", "incompleto", "negativo",
        ))
        if nega:
            return out("false", 0.7)
        if afirm:
            return out("true", 0.7)
        return out("", 0.0, 0.25)

    # Email / correo
    if tipo == "email" or has("email", "correo", "mail"):
        m = _RE_EMAIL.search(texto)
        return out(m.group(0) if m else "", 0.95)
    # Teléfono / celular
    if has("telefono", "teléfono", "celular", "movil", "móvil", "whatsapp", "tel"):
        m = _RE_TEL.search(texto)
        return out(m.group(0) if m else "", 0.9)
    # DNI / cédula / carnet / documento
    if has("dni", "ci", "cedula", "cédula", "carnet", "carnet", "documento"):
        m = _RE_DNI.search(texto)
        return out(m.group(0) if m else "", 0.95)
    # Fecha
    if tipo == "fecha" or has("fecha", "inspeccion", "inspección", "cita", "dia", "día"):
        return out(_extraer_fecha(texto), 0.85, 0.3)
    # Nombre / cliente / solicitante
    if has("nombre", "cliente", "solicitante", "apellido", "apellidos"):
        return out(_extraer_nombre(texto), 0.9, 0.3)
    # Sin heurística para este campo.
    return out("", 0.0, 0.3)
