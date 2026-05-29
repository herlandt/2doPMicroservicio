# Microservicio IA — imagen ligera en modo stub.
# Cuando se sumen TensorFlow / Whisper, cambiar base a python:3.11 (no slim)
# o usar tensorflow/tensorflow:2.17.0 y reinstalar fastapi encima.
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema mínimas (curl para healthchecks, ffmpeg si se conecta Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8001/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
