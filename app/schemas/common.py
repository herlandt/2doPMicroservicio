"""Schemas comunes a varios routers."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    detail: str | None = None


class ReadyResponse(BaseModel):
    ready: bool
    models: dict[str, str]
