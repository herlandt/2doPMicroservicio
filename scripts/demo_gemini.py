r"""Demo / prueba end-to-end del proveedor Gemini (CU-39 dictado + CU-31 asistente).

Sirve para la DEFENSA: sintetiza un dictado HABLADO en español (TTS de Windows),
lo pasa por Gemini 2.5 Flash y verifica que rellena un formulario REAL de la CRE;
luego prueba el asistente. Hace ~2 llamadas a Gemini (cuida el free tier).

Uso (PowerShell, desde ia_service/):
    $env:PYTHONPATH = (Get-Location)
    .\.venv\Scripts\python.exe scripts\demo_gemini.py "TU_API_KEY_1,TU_API_KEY_2"

Requiere: Windows con una voz TTS en español (p. ej. Microsoft Sabina, es-MX).
"""
import os
import subprocess
import sys
import tempfile
import unicodedata

_DICTADO = (
    "Buenas, registro la verificacion. El nombre completo del solicitante es "
    "Juan Perez Mamani. Su numero de cedula de identidad es 7845123. El domicilio "
    "exacto del inmueble es Avenida Busch numero 250, zona Equipetrol. El telefono "
    "de contacto es 70123456. El tipo de conexion solicitada es Trifasica. "
    "Y si, los documentos entregados estan completos."
)

# Formulario REAL "Verificacion Documental" (Backend/.../FormularioSeeder.java)
_CAMPOS = [
    {"nombre": "nombre_solicitante", "tipo": "texto", "etiqueta": "Nombre completo del solicitante"},
    {"nombre": "numero_ci", "tipo": "texto", "etiqueta": "Numero de cedula de identidad"},
    {"nombre": "domicilio", "tipo": "textarea", "etiqueta": "Domicilio exacto del inmueble"},
    {"nombre": "telefono", "tipo": "texto", "etiqueta": "Telefono de contacto"},
    {"nombre": "tipo_solicitud", "tipo": "select", "etiqueta": "Tipo de conexion solicitada",
     "opciones": ["Monofasica", "Trifasica", "Industrial"]},
    {"nombre": "documentos_completos", "tipo": "checkbox", "etiqueta": "Documentos entregados completos"},
]

_ESPERADO = {
    "nombre_solicitante": "Juan Perez Mamani",
    "numero_ci": "7845123",
    "telefono": "70123456",
    "tipo_solicitud": "Trifasica",
    "documentos_completos": "true",
}


def _sintetizar_wav(texto: str) -> str:
    """Genera un WAV hablado en español con el TTS de Windows (SAPI)."""
    wav = os.path.join(tempfile.gettempdir(), "dictado_demo.wav")
    # texto es ASCII sin comillas simples → cadena PS entre comillas simples (robusto).
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try { $s.SelectVoice('Microsoft Sabina Desktop') } catch {};"
        f"$s.SetOutputToWaveFile('{wav}');"
        f"$s.Speak('{texto}');"
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   capture_output=True, timeout=60)
    return wav


def _norm(s: str) -> str:
    """minúsculas, sin espacios y SIN ACENTOS (comparación robusta)."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip().lower().replace(" ", "")


def main() -> None:
    os.environ.setdefault("IA_PROVIDER", "gemini")
    os.environ["GEMINI_API_KEYS"] = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMINI_API_KEYS", "")

    from app import gemini  # import tras fijar el entorno

    print("Sintetizando dictado de voz (es-MX)…")
    wav = _sintetizar_wav(_DICTADO)
    audio = open(wav, "rb").read()
    print(f"  WAV: {wav} ({len(audio)} bytes)\n")

    print("=" * 70)
    print("PRUEBA 1 — DICTADO POR VOZ → Formulario 'Verificacion Documental'")
    print("=" * 70)
    texto, sug = gemini.extraer_de_audio(audio, "audio/wav", _CAMPOS)
    print("Transcripción de Gemini:\n  " + texto + "\n")
    obtenido = {s.campo: s.valor for s in sug}
    print("Campos rellenados por la IA:")
    for c in _CAMPOS:
        n = c["nombre"]
        print(f"  - {n:22} = {obtenido.get(n, '(vacío)')!r}")

    print("\nVerificación (campos clave):")
    ok = 0
    for k, esperado in _ESPERADO.items():
        got = obtenido.get(k, "")
        passed = bool(got) and (_norm(esperado) in _norm(got) or _norm(got) in _norm(esperado))
        ok += passed
        print(f"  [{'PASS' if passed else 'REVISAR'}] {k}: esperado≈{esperado!r} | obtenido={got!r}")
    print(f"\n  >>> {ok}/{len(_ESPERADO)} campos clave correctos.")

    print("\n" + "=" * 70)
    print("PRUEBA 2 — ASISTENTE (cuando TensorFlow no resuelve → Gemini)")
    print("=" * 70)
    ctx = ("Rol del usuario: CLIENTE. Tramites disponibles: Conexion electrica nueva, "
           "Cambio de medidor, Verificacion documental.")
    preg = "Quiero poner luz en una casa nueva que estoy construyendo, ¿qué trámite necesito y qué documentos?"
    print("Pregunta:", preg)
    print("Respuesta de Gemini:\n  " + gemini.responder_asistente(preg, ctx))


if __name__ == "__main__":
    main()
