# Microservicio IA — con TensorFlow (clasificador de intención CU-46).
# Base no-slim: trae las libs de sistema (libgomp, etc.) que necesita TF.
FROM python:3.11

WORKDIR /app

# curl para healthchecks; libgomp1 lo usa TensorFlow (OpenMP); ffmpeg convierte
# el audio webm del navegador a wav 16k para Gemini (CU-39 dictado por voz).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8001/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
