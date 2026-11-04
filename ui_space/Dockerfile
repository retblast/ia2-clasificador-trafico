FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py inference.py ./
COPY salidas_seguridad_sin_puerto ./salidas_seguridad_sin_puerto
COPY salidas_aplicaciones ./salidas_aplicaciones
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
