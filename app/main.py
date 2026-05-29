"""Microservicio IA — Parte 2 del proyecto de trámites.

Esqueleto FastAPI con stubs deterministas. Cuando se conecten los modelos
TensorFlow reales, cada router individual los carga sin tocar este archivo.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import asignacion, enrutamiento, health, nlp, reportes

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger(__name__)


class UTF8JSONResponse(JSONResponse):
    """Fuerza ``Content-Type: application/json; charset=utf-8`` para evitar
    que clientes estrictos (Spring RestClient en Windows) caigan en
    ISO-8859-1 al deserializar bytes UTF-8."""
    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title="IA Service - Trámites Parte 2",
    description="Microservicio Python/FastAPI para NLP + clasificación + enrutamiento.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=UTF8JSONResponse,
)

# Solo el backend Spring debe llegar — en prod cierra el CORS o usa firewall.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(nlp.router)
app.include_router(asignacion.router)
app.include_router(reportes.router)
app.include_router(enrutamiento.router)


@app.on_event("startup")
def on_startup() -> None:
    log.info("IA Service iniciado (stub mode)")
    log.info("Mongo URI: %s", _redact_uri(settings.mongo_uri))
    log.info("Modelos path: %s", settings.models_path)
    log.info("Whisper model: %s", settings.whisper_model)


def _redact_uri(uri: str) -> str:
    """Oculta password del URI para no filtrarlo en logs."""
    if "@" not in uri:
        return uri
    user_pass, host = uri.rsplit("@", 1)
    if "://" in user_pass:
        scheme, _ = user_pass.split("://", 1)
        return f"{scheme}://***:***@{host}"
    return f"***:***@{host}"
