"""Proveedor Gemini (Google AI Studio) — CU-39 dictado + CU-31/46 asistente.

Gemini 2.5 Flash es multimodal: en UNA sola llamada transcribe el audio del
funcionario Y rellena el formulario contra un ``responseSchema`` JSON. Es gratis
con cuentas de AI Studio; se ROTAN varias API keys para multiplicar la cuota del
free tier (al toparse con 429/403 se prueba la siguiente cuenta).

Sin SDK: REST directo con ``urllib`` (stdlib), así no añade dependencias. Cada
función LANZA ante cualquier fallo para que el llamador degrade al heurístico /
KB local sin romper la demo.
"""
import base64
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request

from app.config import settings
from app.schemas.nlp import CampoSugerido

log = logging.getLogger(__name__)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_RESERVADO = "_transcripcion"  # campo interno donde Gemini deja la transcripción literal


def _keys() -> list[str]:
    return [k.strip() for k in (settings.gemini_api_keys or "").split(",") if k.strip()]


def _post(parts: list, generation_config: dict | None = None) -> dict:
    """POST a Gemini. Rota a la siguiente API key al toparse con la CUOTA
    (429/403); reintenta la MISMA key en errores TRANSITORIOS (500/503/504, red,
    timeout) con backoff. LANZA si todo falla."""
    keys = _keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEYS vacío")
    body: dict = {"contents": [{"parts": parts}]}
    if generation_config:
        body["generationConfig"] = generation_config
    data = json.dumps(body).encode()
    url = _ENDPOINT.format(model=settings.gemini_model)

    ultimo = "sin intentos"
    for i, key in enumerate(keys):
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        for intento in range(3):
            try:
                req = urllib.request.Request(url, data=data, method="POST", headers=headers)
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                cuerpo = ""
                try:
                    cuerpo = e.read().decode()[:200]
                except Exception:  # noqa: BLE001
                    pass
                ultimo = f"HTTP {e.code}"
                if e.code in (429, 403):  # cuota agotada → probar otra cuenta
                    log.warning("Gemini key #%d sin cuota (%s); roto a la siguiente", i + 1, e.code)
                    break
                if e.code in (500, 503, 504) and intento < 2:  # transitorio → reintentar
                    log.warning("Gemini %s (intento %d/3), reintento…", e.code, intento + 1)
                    time.sleep(2 * (intento + 1))
                    continue
                raise RuntimeError(f"Gemini HTTP {e.code}: {cuerpo}")
            except Exception as e:  # noqa: BLE001 — red/timeout transitorio
                ultimo = str(e)
                if intento < 2:
                    time.sleep(2 * (intento + 1))
                    continue
                break
    raise RuntimeError(f"Gemini: todas las keys fallaron ({ultimo})")


def _texto_de(resp: dict) -> str:
    for cand in resp.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "text" in p:
                return p["text"]
    return ""


def _a_wav(audio: bytes, mime: str | None) -> tuple[bytes, str]:
    """Convierte el audio a WAV 16 kHz mono con ffmpeg (formato validado en
    Gemini). Si no hay ffmpeg o falla, devuelve el audio original con su mime
    para intentarlo directo (Gemini suele aceptar ogg/mp4 también)."""
    if mime and "wav" in mime:
        return audio, "audio/wav"
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
            input=audio, capture_output=True, timeout=30,
        )
        if p.returncode == 0 and p.stdout:
            return p.stdout, "audio/wav"
        log.warning("ffmpeg devolvió %s; envío el audio original", p.returncode)
    except FileNotFoundError:
        log.warning("ffmpeg no instalado; envío el audio original a Gemini")
    except Exception as e:  # noqa: BLE001
        log.warning("Conversión de audio falló (%s); envío el original", e)
    return audio, (mime or "audio/webm")


def _schema(campos: list) -> dict:
    """JSON Schema que fuerza la salida de Gemini contra el formulario. Incluye un
    campo reservado para la transcripción literal. Sin 'required' → solo rellena
    lo presente."""
    props: dict = {
        _RESERVADO: {"type": "string", "description": "Transcripción literal de lo dictado"}
    }
    for c in campos:
        nombre = c.get("nombre")
        if not nombre:
            continue
        tipo = (c.get("tipo") or "texto").lower()
        prop: dict = {"type": "string", "description": c.get("etiqueta") or nombre}
        opciones = c.get("opciones") or []
        if tipo == "checkbox":
            prop["enum"] = ["true", "false"]
        elif tipo in ("select", "radio") and opciones:
            # radio = opción única, mismo dominio cerrado que un select.
            prop["enum"] = [str(o) for o in opciones]
        props[nombre] = prop
    return {"type": "object", "properties": props}


_INSTRUCCION_DICTADO = (
    "Eres un asistente que transcribe el audio dictado por un funcionario y "
    "rellena el formulario de un trámite. Rellena SOLO los campos mencionados en "
    "el audio; deja en cadena vacía los que no se mencionen (no inventes). Fechas "
    "en formato dd/mm/aaaa. Números (cédula, teléfono, montos) en dígitos, sin "
    "separadores. Casillas: 'true' o 'false'. Listas: exactamente una de las "
    f"opciones permitidas. Pon la transcripción literal en el campo '{_RESERVADO}'."
)


def extraer_de_audio(audio: bytes, mime: str | None, campos: list) -> tuple[str, list[CampoSugerido]]:
    """Una sola llamada multimodal: audio → (transcripción, campos rellenos)."""
    wav, real_mime = _a_wav(audio, mime)
    parts = [
        {"inline_data": {"mime_type": real_mime, "data": base64.b64encode(wav).decode()}},
        {"text": _INSTRUCCION_DICTADO},
    ]
    resp = _post(parts, {
        "responseMimeType": "application/json",
        "responseSchema": _schema(campos),
        "maxOutputTokens": 2048,
        "temperature": 0,
    })
    raw = _texto_de(resp)
    datos = json.loads(raw) if raw.strip() else {}

    transcripcion = str(datos.pop(_RESERVADO, "") or "")
    sugerencias: list[CampoSugerido] = []
    for nombre, valor in datos.items():
        if valor is None:
            continue
        v = str(valor).strip()
        if not v:
            continue
        sugerencias.append(CampoSugerido(campo=nombre, valor=v, confianza=0.9))
    return transcripcion, sugerencias


_SISTEMA_ASISTENTE = (
    "Eres el asistente del Sistema de Gestión de Trámites de la CRE (cooperativa "
    "eléctrica boliviana). Respondes SOLO sobre cómo usar el sistema (iniciar y "
    "seguir trámites, documentos, expediente, flujos, métricas, notificaciones). "
    "Sé breve (2-4 frases), claro y en español. Usa el CONTEXTO que se te da; si "
    "la pregunta no es sobre el sistema, dilo amablemente y ofrece en qué sí "
    "puedes ayudar. No inventes datos de trámites que no estén en el contexto."
)


def responder_asistente(consulta: str, contexto: str = "") -> str:
    """Respuesta generativa del asistente (CU-31) para los casos que TensorFlow
    no resuelve. LANZA si Gemini falla → el backend degrada a su KB local."""
    prompt = _SISTEMA_ASISTENTE + "\n\n"
    if contexto:
        prompt += f"CONTEXTO DEL SISTEMA:\n{contexto}\n\n"
    prompt += f"PREGUNTA DEL USUARIO:\n{consulta}"
    resp = _post([{"text": prompt}], {"maxOutputTokens": 1024, "temperature": 0.2})
    texto = _texto_de(resp).strip()
    if not texto:
        raise RuntimeError("Gemini no devolvió texto")
    return texto


# ── Diseño de flujo por prompt (CU-14) ───────────────────────────────────────
_SISTEMA_FLUJO = (
    "Eres un modelador de procesos. A partir de la DESCRIPCION en lenguaje natural de un "
    "proceso/tramite, genera un diagrama de actividad UML con swimlanes. Reglas: usa SOLO los "
    "departamentos disponibles como swimlanes (campo 'departamento' de cada actividad, con el "
    "nombre EXACTO de la lista dada); incluye un nodo 'inicio' (primero) y un nodo 'fin' (ultimo); "
    "cada paso del proceso es un nodo 'actividad' con su departamento responsable; si el texto "
    "describe pasos SIMULTANEOS usa un 'fork' antes y un 'join' despues, conectando en paralelo "
    "SOLO esos pasos (el resto va lineal); si hay una condicion (aprobar/rechazar/decidir) usa un "
    "nodo 'decision' con dos transiciones etiquetadas 'si' y 'no'; si el texto describe volver a un "
    "paso anterior (reproceso/observacion) usa una transicion 'iterativo' hacia ese nodo; marca "
    "opcional=true en los pasos que el texto indique como opcionales. Las transiciones referencian "
    "los nodos por su INDICE (0-based) en el arreglo 'nodos'. No inventes departamentos fuera de la lista."
)

_SCHEMA_FLUJO = {
    "type": "object",
    "properties": {
        "nodos": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "tipo": {"type": "string", "enum": ["inicio", "actividad", "decision", "fork", "join", "fin"]},
                "nombre": {"type": "string"},
                "departamento": {"type": "string"},
                "opcional": {"type": "boolean"},
            },
            "required": ["tipo", "nombre"],
        }},
        "transiciones": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "origen": {"type": "integer"},
                "destino": {"type": "integer"},
                "tipo": {"type": "string", "enum": ["secuencial", "paralelo", "condicional", "iterativo"]},
                "etiqueta": {"type": "string"},
            },
            "required": ["origen", "destino", "tipo"],
        }},
    },
    "required": ["nodos", "transiciones"],
}


def generar_flujo(prompt: str, departamentos: list[str]) -> dict:
    """NL → diagrama de actividad (nodos + transiciones). Modelo multimodal con
    responseSchema: comprende sinónimos, topología mixta y qué pasos van en
    paralelo. LANZA si Gemini falla → el backend usa su heurística local."""
    deps = ", ".join(d for d in departamentos if d) or "(ninguno)"
    texto = f"{_SISTEMA_FLUJO}\n\nDEPARTAMENTOS DISPONIBLES: {deps}\n\nDESCRIPCION DEL PROCESO:\n{prompt}"
    resp = _post([{"text": texto}], {
        "responseMimeType": "application/json",
        "responseSchema": _SCHEMA_FLUJO,
        "maxOutputTokens": 2048,
        "temperature": 0.2,
    })
    raw = _texto_de(resp)
    datos = json.loads(raw) if raw.strip() else {}
    if not datos.get("nodos"):
        raise RuntimeError("Gemini no devolvió un flujo válido")
    return datos
