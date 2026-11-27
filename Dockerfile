# Imagen de SpringPlot
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Código
COPY backend backend
COPY frontend frontend

# Usuario sin privilegios
RUN useradd --create-home springplot
USER springplot

WORKDIR /app/backend
EXPOSE 8000
# Puerto de la variable PORT (8000 por defecto)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
