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
