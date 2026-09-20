# Imagen del servicio de predicción.
# El contexto de build es la raíz del repositorio:
#   docker build -f deployment/Dockerfile -t fraud-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_URI=/app/models/champion

WORKDIR /app

# El modelo campeón exportado trae su propio requirements.txt generado por
# MLflow: instalarlo garantiza las mismas versiones con las que se entrenó.
COPY models/champion/ ./models/champion/
COPY deployment/requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir -r models/champion/requirements.txt \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
