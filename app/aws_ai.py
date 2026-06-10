"""IA gestionada en AWS — CU-39 (dictado voz→formulario) y CU-31/46 (asistente).

Dos servicios:
  • Amazon Transcribe  → voz a texto real (reemplaza el texto stub).
  • Amazon Bedrock (Claude Haiku) → extracción estructurada de campos y
    respuesta del asistente para los casos que TensorFlow no resuelve.

boto3 se importa de forma PEREZOSA y los clientes se cachean: si AWS_AI_ENABLED
está en False o boto3 no está instalado, este módulo ni se toca. Cada función
LANZA excepción ante cualquier fallo (sin credenciales, sin acceso al modelo,
timeout…) para que el llamador degrade al modo local/heurístico sin romperse.

Credenciales: en prod corre en la EC2 con rol IAM (IMDSv2) → boto3 las resuelve
solo, sin claves. El rol necesita: transcribe:*, s3 (sobre el bucket temporal) y
bedrock:InvokeModel sobre el modelo Claude habilitado.
"""
import json
import logging
import time
import uuid
from functools import lru_cache

from app.config import settings
from app.schemas.nlp import CampoSugerido

log = logging.getLogger(__name__)


# ── Clientes boto3 (perezosos y cacheados) ───────────────────────────────────
@lru_cache(maxsize=1)
def _s3():
    import boto3
    return boto3.client("s3", region_name=settings.aws_region)


@lru_cache(maxsize=1)
def _transcribe():
    import boto3
    return boto3.client("transcribe", region_name=settings.aws_region)


@lru_cache(maxsize=1)
def _bedrock():
    import boto3
    return boto3.client("bedrock-runtime", region_name=settings.aws_region)


# ── Amazon Transcribe: voz → texto ───────────────────────────────────────────
def _ext_desde(filename: str | None, content_type: str | None) -> str:
    """Formato de medio para Transcribe (mp3|mp4|wav|flac|ogg|amr|webm)."""
    ct = (content_type or "").lower()
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in ("mp3", "mp4", "m4a", "wav", "flac", "ogg", "amr", "webm"):
            return "mp4" if ext == "m4a" else ext
    if "webm" in ct:
        return "webm"
    if "ogg" in ct:
        return "ogg"
    if "wav" in ct:
        return "wav"
    if "mp4" in ct or "m4a" in ct:
        return "mp4"
    if "mpeg" in ct or "mp3" in ct:
        return "mp3"
    # MediaRecorder del navegador entrega webm por defecto.
    return "webm"


def transcribir(audio_bytes: bytes, filename: str | None, content_type: str | None) -> str:
    """Sube el audio a S3, lanza un job de Transcribe, espera el resultado y
    devuelve el texto. Limpia los temporales. LANZA si falla o agota el tiempo."""
    bucket = settings.aws_s3_bucket
    ext = _ext_desde(filename, content_type)
    job = "dictado-" + uuid.uuid4().hex
    in_key = f"{settings.transcribe_prefix}{job}.{ext}"
    out_key = f"{settings.transcribe_prefix}{job}.json"

    s3 = _s3()
    transcribe = _transcribe()

    s3.put_object(Bucket=bucket, Key=in_key, Body=audio_bytes)
    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job,
            LanguageCode=settings.transcribe_language,
            MediaFormat=ext,
            Media={"MediaFileUri": f"s3://{bucket}/{in_key}"},
            OutputBucketName=bucket,
            OutputKey=out_key,
        )
        deadline = time.time() + settings.transcribe_timeout_s
        estado = "IN_PROGRESS"
        while time.time() < deadline:
            job_info = transcribe.get_transcription_job(TranscriptionJobName=job)["TranscriptionJob"]
            estado = job_info["TranscriptionJobStatus"]
            if estado in ("COMPLETED", "FAILED"):
                break
            time.sleep(2)
        if estado != "COMPLETED":
            raise RuntimeError(f"Transcribe no completó a tiempo (estado={estado})")

        obj = s3.get_object(Bucket=bucket, Key=out_key)
        data = json.loads(obj["Body"].read())
        return data["results"]["transcripts"][0]["transcript"]
    finally:
        # Limpieza best-effort: no dejar audios ni jobs colgando.
        try:
            s3.delete_object(Bucket=bucket, Key=in_key)
            s3.delete_object(Bucket=bucket, Key=out_key)
            transcribe.delete_transcription_job(TranscriptionJobName=job)
        except Exception:  # noqa: BLE001 — limpieza, no debe enmascarar el resultado
            pass


# ── Amazon Bedrock: extracción de campos (CU-39) ─────────────────────────────
def _input_schema(campos: list) -> dict:
    """Construye un JSON Schema para forzar la salida del LLM contra el formulario.
    Cada campo es opcional (sin 'required') para que el modelo solo rellene lo que
    el texto menciona. select → enum de opciones; checkbox → 'true'/'false'."""
    props: dict = {}
    for c in campos:
        nombre = c.get("nombre")
        if not nombre:
            continue
        tipo = (c.get("tipo") or "texto").lower()
        etiqueta = c.get("etiqueta") or nombre
        opciones = c.get("opciones") or []
        prop: dict = {"type": "string", "description": etiqueta}
        if tipo == "checkbox":
            prop["enum"] = ["true", "false"]
        elif tipo == "select" and opciones:
            prop["enum"] = [str(o) for o in opciones]
        props[nombre] = prop
    return {"type": "object", "properties": props, "additionalProperties": False}


_SISTEMA_EXTRACCION = (
    "Eres un asistente que rellena el formulario de un trámite a partir de lo que "
    "DICTÓ un funcionario. Extrae únicamente los datos presentes en el texto y "
    "colócalos en el campo correspondiente usando la herramienta. Reglas: no "
    "inventes; deja fuera (no incluyas) cualquier campo que el texto no mencione; "
    "las fechas en formato dd/mm/aaaa; los campos de tipo casilla devuelven "
    "'true' o 'false'; los de lista deben coincidir EXACTAMENTE con una de las "
    "opciones permitidas."
)


def extraer_campos(campos: list, texto: str) -> list[CampoSugerido]:
    """Pide a Bedrock que rellene el formulario contra su esquema (tool use).
    Devuelve solo los campos que el modelo rellenó. LANZA si Bedrock falla."""
    tool = {
        "toolSpec": {
            "name": "rellenar_formulario",
            "description": "Coloca cada dato extraído del dictado en su campo del formulario.",
            "inputSchema": {"json": _input_schema(campos)},
        }
    }
    resp = _bedrock().converse(
        modelId=settings.bedrock_model_id,
        system=[{"text": _SISTEMA_EXTRACCION}],
        messages=[{
            "role": "user",
            "content": [{"text": f'Texto dictado:\n"""{texto}"""\n\nRellena el formulario con los datos presentes.'}],
        }],
        toolConfig={
            "tools": [tool],
            "toolChoice": {"tool": {"name": "rellenar_formulario"}},
        },
        inferenceConfig={"maxTokens": 1024, "temperature": 0},
    )
    bloques = resp.get("output", {}).get("message", {}).get("content", [])
    datos: dict = {}
    for b in bloques:
        if "toolUse" in b and b["toolUse"].get("name") == "rellenar_formulario":
            datos = b["toolUse"].get("input", {}) or {}
            break

    sugerencias: list[CampoSugerido] = []
    for nombre, valor in datos.items():
        if valor is None:
            continue
        val = str(valor).strip()
        if not val:
            continue
        sugerencias.append(CampoSugerido(campo=nombre, valor=val, confianza=0.9))
    return sugerencias


# ── Amazon Bedrock: respuesta del asistente (CU-31/46 híbrido) ───────────────
_SISTEMA_ASISTENTE = (
    "Eres el asistente del Sistema de Gestión de Trámites de la CRE (cooperativa "
    "eléctrica boliviana). Respondes SOLO sobre cómo usar el sistema (iniciar y "
    "seguir trámites, documentos, expediente, flujos, métricas, notificaciones). "
    "Sé breve (2-4 frases), claro y en español. Usa el CONTEXTO que se te da; si "
    "la pregunta no es sobre el sistema, dilo amablemente y ofrece en qué sí puedes "
    "ayudar. No inventes datos de trámites que no estén en el contexto."
)


def responder_asistente(consulta: str, contexto: str = "") -> str:
    """Respuesta generativa del asistente para los casos que TensorFlow no
    resuelve (baja confianza / fuera de alcance). LANZA si Bedrock falla."""
    user = consulta if not contexto else f"CONTEXTO DEL SISTEMA:\n{contexto}\n\nPREGUNTA DEL USUARIO:\n{consulta}"
    resp = _bedrock().converse(
        modelId=settings.bedrock_model_id,
        system=[{"text": _SISTEMA_ASISTENTE}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.2},
    )
    bloques = resp.get("output", {}).get("message", {}).get("content", [])
    for b in bloques:
        if "text" in b and b["text"].strip():
            return b["text"].strip()
    raise RuntimeError("Bedrock no devolvió texto")
