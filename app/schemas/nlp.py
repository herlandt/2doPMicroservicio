"""CU-39 — voz a formulario."""
from pydantic import BaseModel, Field


class CampoSchema(BaseModel):
    """Definición de un campo del formulario activo que el mapeo debe rellenar."""
    nombre: str
    tipo: str = Field(description="texto|numero|fecha|select|checkbox|textarea")
    validaciones: dict | None = None


class CampoSugerido(BaseModel):
    campo: str
    valor: str
    confianza: float


class VozAFormularioResponse(BaseModel):
    texto_transcrito: str
    campos: list[CampoSugerido]
