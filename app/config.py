"""Configuración del microservicio IA. Variables de entorno con defaults seguros."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Mongo (Parte 2 — el microservicio puede leer histórico para entrenar)
    mongo_uri: str = "mongodb://admin:12345678@localhost:27017/tramites_db?authSource=admin"

    # S3 / AWS (para guardar audios de transcripción)
    aws_region: str = "us-east-1"
    aws_s3_bucket: str = "tramites-dev"

    # Directorio de modelos entrenados (.h5 / SavedModel) — se carga al startup
    models_path: str = "./models_artifacts"

    # Whisper para transcripción de voz (CU-39, CU-40)
    # Valores: tiny | base | small | medium | large.
    # Para CPU recomendamos "small"; para demo "tiny" basta.
    whisper_model: str = "tiny"

    # ── IA gestionada en AWS (CU-39 dictado + CU-31/46 asistente híbrido) ──
    # Si es False (default), TODO degrada al modo local/heurístico: el proyecto
    # sigue funcionando sin AWS y sin gastar. Poner True en prod (EC2 con rol IAM)
    # para activar Amazon Transcribe (STT) + Amazon Bedrock (extracción/respuesta).
    aws_ai_enabled: bool = False
    # ID del modelo Claude en Bedrock. VERIFICADO en la cuenta 217517350542
    # (us-east-1): Haiku 4.5 existe solo vía PERFIL DE INFERENCIA "us." — el ID
    # base "anthropic.claude-haiku-4-5-20251001-v1:0" NO se puede invocar on-demand.
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # Idioma para Amazon Transcribe (es-US | es-ES).
    transcribe_language: str = "es-US"
    # Prefijo en el bucket S3 (aws_s3_bucket) para los audios temporales de Transcribe.
    transcribe_prefix: str = "transcribe-tmp/"
    # Segundos máximos a esperar a que Transcribe termine antes de degradar al stub.
    transcribe_timeout_s: int = 45

    # ── Proveedor de IA externa: local | gemini | aws ──
    # local  → heurística/stub (sin red, $0).
    # gemini → Google AI Studio (Gemini 2.5 Flash): multimodal, hace dictado y
    #          asistente; gratis con cuentas de AI Studio (se rotan varias keys).
    # aws    → Amazon Transcribe + Bedrock (requiere acceso al modelo en Bedrock).
    ia_provider: str = "local"
    # API keys de Gemini SEPARADAS POR COMA. Se rotan al toparse con la cuota del
    # free tier (429/403), multiplicando el cupo con varias cuentas. NO commitear.
    gemini_api_keys: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Tamaño máximo permitido para el audio de voz-a-formulario (CU-39).
    # Si la petición lo excede devolvemos HTTP 413. Default: 25 MB.
    max_audio_bytes: int = 25 * 1024 * 1024

    log_level: str = "INFO"

    # Orígenes permitidos para CORS. Vacío => default seguro en main.py.
    cors_origins: list[str] = []

    # Si está definido, el microservicio valida que las peticiones traigan
    # este JWT compartido con Spring para evitar acceso directo de clientes.
    # En MVP lo dejamos vacío y confiamos en red privada / firewall.
    backend_shared_secret: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()
