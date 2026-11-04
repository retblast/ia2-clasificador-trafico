FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py inference.py ./
COPY salidas_seguridad_sin_puerto ./salidas_seguridad_sin_puerto
COPY salidas_aplicaciones ./salidas_aplicaciones
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
