"""CU-39 — voz a formulario."""
from pydantic import BaseModel, Field


class CampoSchema(BaseModel):
    """Definición de un campo del formulario activo que el mapeo debe rellenar."""
    nombre: str
    tipo: str = Field(description="texto|numero|fecha|select|checkbox|textarea")
    # Etiqueta humana (aporta señal extra al matching) y opciones de un 'select'.
    # Opcionales para no romper a quien envíe el schema mínimo (solo nombre/tipo).
    etiqueta: str | None = None
    opciones: list[str] | None = None
    validaciones: dict | None = None


class CampoSugerido(BaseModel):
    campo: str
    valor: str
    confianza: float


class VozAFormularioResponse(BaseModel):
    texto_transcrito: str
    campos: list[CampoSugerido]
