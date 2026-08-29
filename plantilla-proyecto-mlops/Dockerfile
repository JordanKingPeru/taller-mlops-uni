# Imagen del servicio de predicción — multi-etapa simple
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias primero (aprovecha la caché de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código, configuración y el modelo entrenado
COPY params.yaml .
COPY src/ src/
COPY models/ models/

EXPOSE 8000
# Usuario no-root: buena práctica de seguridad
RUN useradd -m apiuser && chown -R apiuser /app
USER apiuser

CMD ["uvicorn", "src.servicio.app:app", "--host", "0.0.0.0", "--port", "8000"]
