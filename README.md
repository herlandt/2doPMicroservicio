# IA Service — Microservicio Python/FastAPI

Microservicio de la Parte 2. Expone 7 endpoints que Spring Boot consume
mediante el `IaProxyService`.

**Estado actual:** *stub mode* — todos los endpoints devuelven respuestas
deterministas plausibles para que Spring + el frontend funcionen end-to-end
sin necesidad de tener modelos entrenados todavía. Los modelos TensorFlow
reales se enchufan endpoint por endpoint cuando estén listos.

## Endpoints

| Método | Ruta | CU | Descripción |
|--------|------|----|-------------|
| GET  | `/healthz` | — | Liveness |
| GET  | `/readyz`  | — | Readiness + estado de modelos |
| POST | `/nlp/voz-a-formulario` | CU-39 | Voz → campos del formulario |
| POST | `/asignacion/politica` | CU-40 | Descripción → top 3 políticas |
| POST | `/reportes/consulta-natural` | CU-41 | Consulta NL → pipeline Mongo |
| POST | `/enrutamiento/ruta-optima` | CU-42 | Sugerencia de ruta del trámite |
| POST | `/enrutamiento/riesgo-demora` | CU-43 | Predicción de SLA (batch) |
| POST | `/enrutamiento/prioridades` | CU-44 | Ordena bandeja del funcionario |
| POST | `/enrutamiento/anomalias` | CU-45 | Detección de outliers |
| POST | `/enrutamiento/modelos/reentrenar` | — | Reentrenar (stub) |

Swagger UI: `http://localhost:8001/docs`.

## Arrancar en local (Windows / PowerShell)

```powershell
cd ..\ia_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Verificar:
```powershell
curl http://localhost:8001/healthz
curl http://localhost:8001/readyz
```

## Arrancar con Docker

```bash
docker build -t ia-service .
docker run --rm -p 8001:8001 ia-service
```

## Configuración (variables de entorno)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://admin:12345678@localhost:27017/tramites_db?authSource=admin` | Usado al cargar histórico para entrenamiento |
| `AWS_REGION` | `us-east-1` | Para subir audios a S3 |
| `AWS_S3_BUCKET` | `tramites-dev` | |
| `MODELS_PATH` | `./models_artifacts` | Donde se cargan los `.h5`/SavedModel |
| `WHISPER_MODEL` | `tiny` | tiny\|base\|small\|medium\|large |
| `LOG_LEVEL` | `INFO` | |
| `BACKEND_SHARED_SECRET` | (vacío) | JWT compartido para validar peticiones de Spring |

## Próximos pasos

1. Conectar **Whisper** real en `/nlp/voz-a-formulario` (CU-39).
2. Entrenar el **clasificador de política** (CU-40) con `sugerencias_politica.feedback=ACEPTADA` del histórico.
3. Entrenar el modelo de **riesgo SLA** (CU-43) con `metricas_tiempo` + `tramites` cerrados.
4. Reemplazar la heurística de **anomalías** (CU-45) por un autoencoder o `IsolationForest`.
