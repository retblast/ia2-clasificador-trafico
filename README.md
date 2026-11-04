
# Prototipo de clasificación de tráfico de red

El paquete contiene una interfaz Streamlit y una API FastAPI que comparten la misma lógica de validación e inferencia.

## Estructura esperada

```text
.
├── app.py
├── api.py
├── inference.py
├── train_validate.py
├── requirements.txt
├── tests/
└── salidas_seguridad_sin_puerto/
    ├── modelo_entrenado.joblib
    ├── caracteristicas.json
    └── etiquetas.json
└── salidas_aplicaciones/
    ├── modelo_entrenado.joblib
    ├── caracteristicas.json
    └── etiquetas.json
```

El paquete incluye los artefactos reales generados por `train_validate.py` en ambos directorios. También incluye las matrices de prueba reconstruidas y sus predicciones esperadas para ejecutar pruebas de paridad. Los modelos serializados registran scikit-learn 1.9.0, por lo que `requirements.txt` fija esa versión. El programa valida que el modelo de seguridad tenga 68 características y 2 clases, y que el modelo de aplicaciones tenga 23 características y 7 clases. No se incluyen modelos ficticios.

## Instalar y ejecutar localmente

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

En Windows, activar el entorno con `.venv\Scripts\activate`.

La UI permite seleccionar el modelo, consultar el esquema de entrada, cargar CSV, revisar errores y advertencias, ver conteos por clase, visualizar un gráfico de barras y descargar las predicciones.

### Validación de CSV

- Las columnas requeridas deben existir.
- Los valores no numéricos no vacíos se rechazan.
- Los valores infinitos se rechazan.
- Los vacíos/NaN se notifican y se permiten solo si el pipeline cargado contiene un paso de imputación.
- Las columnas adicionales se conservan en el resultado, pero no se pasan al modelo.
- El orden de las características se fuerza según `caracteristicas.json`.

## API local

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Comprobar estado:

```bash
curl http://127.0.0.1:8000/health
```

Inferencia:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "model=seguridad" \
  -F "file=@muestra_seguridad.csv"
```

Para el otro clasificador, cambiar `model=seguridad` por `model=aplicaciones`.

`GET /health` informa `ok` cuando ambos modelos están disponibles y `degraded` cuando falta al menos uno. Esto evita declarar saludable un servicio que no puede atender ambas tareas.

## Pruebas

El script de entrenamiento ahora exporta `caracteristicas_prueba.csv`, con las mismas filas y el mismo orden que `predicciones_prueba.csv`. Esto es necesario porque la versión previa del script solo exportaba las etiquetas reales y predichas, sin los vectores de características que permiten repetir la inferencia.

Después de volver a ejecutar el entrenamiento para cada experimento:

```bash
pytest -q
```

Las pruebas de paridad real se activan cuando están presentes, dentro de cada carpeta de salida:

- `modelo_entrenado.joblib`
- `caracteristicas.json`
- `etiquetas.json`
- `caracteristicas_prueba.csv`
- `predicciones_prueba.csv`

Sin estos archivos, las pruebas de integración se omiten explícitamente. No se deben interpretar como aprobadas.

## Despliegue recomendado: dos Hugging Face Docker Spaces

Se prepararon dos espacios independientes con SDK Docker, uno para la interfaz y otro para la API. Esta opción evita ejecutar dos procesos de servicio dentro de un mismo contenedor.

**Importante sobre compatibilidad:** los modelos serializados reportan haber sido creados con scikit-learn 1.9.0. Los tres archivos `requirements.txt` fijan `scikit-learn==1.9.0`; no cambies esa versión sin volver a validar las predicciones.

### A. Space de interfaz (Streamlit sobre Docker)

1. Crea un Space en `https://huggingface.co/new-space`.
2. Selecciona SDK `Docker`.
3. Clona el repositorio del Space.
4. Copia el contenido de `ui_space/` a la raíz del repositorio.
5. Conserva el `README.md` con metadatos `sdk: docker` y el `Dockerfile`.
6. Copia las carpetas `salidas_seguridad_sin_puerto/` y `salidas_aplicaciones/` con sus tres artefactos obligatorios.
7. Sube los cambios. Para los `.joblib`, usa Git LFS:
   ```bash
   git lfs install
   git lfs track "*.joblib"
   git add .gitattributes
   git add .
   git commit -m "Deploy Streamlit traffic classifier"
   git push
   ```
8. Espera a que el Space termine la construcción y muestre `Running`.
9. Abre la URL asignada por Hugging Face y prueba ambos modelos con los CSV de características de prueba.
10. Guarda la URL real, capturas y los CSV descargados como evidencia.

### B. Space de API (FastAPI)

1. Crea otro Space en `https://huggingface.co/new-space`.
2. Selecciona SDK `Docker`.
3. Clona el repositorio.
4. Copia el contenido de `api_space/` a la raíz del repositorio, incluyendo el `README.md` con metadatos Docker y el `Dockerfile`.
5. Copia las carpetas `salidas_seguridad_sin_puerto/` y `salidas_aplicaciones/` con sus tres artefactos obligatorios.
6. Sube los cambios usando Git LFS para los `.joblib` si corresponde.
7. Espera el estado `Running`.
8. Comprueba `https://<tu-space-api>.hf.space/health`.
9. Prueba `POST /predict` mediante `curl` o la documentación `/docs`.
10. Conserva las respuestas HTTP reales como evidencia.

No copies literalmente una URL de ejemplo: usa únicamente las URLs asignadas a tus Spaces.

## Evidencia que debe recolectarse manualmente

- URL pública de la UI abierta y operativa.
- Captura de la selección del modelo de seguridad con sus 68 características.
- Captura de la selección del modelo de aplicaciones con sus 23 características.
- Captura de una predicción real para cada modelo.
- Captura de la tabla de conteos y gráfico.
- Archivo CSV descargado desde la UI.
- Salida completa de `pytest -q`.
- Respuesta real de `GET /health`.
- Respuesta real de `POST /predict` para ambos modelos.
- Fecha de prueba y versión/commit desplegado.

No se afirma que exista despliegue, URL pública, captura o ejecución de pruebas hasta que el responsable realice estos pasos y conserve los resultados.

## Límites

Los modelos se entrenaron con benchmarks: CICIDS2017 e ISCXVPN2016. Las predicciones sobre otros entornos pueden degradarse. Una etiqueta `Attack` no constituye por sí sola prueba de una intrusión y las categorías de aplicación son estimaciones estadísticas.


## Verificación local realizada con los artefactos recibidos

- Paridad exacta: 5.957/5.957 predicciones CICIDS2017 y 2.940/2.940 predicciones ISCXVPN2016 coinciden con `predicciones_prueba.csv`.
- API local mediante FastAPI TestClient: `GET /health` respondió HTTP 200 y estado `ok`; `POST /predict` respondió HTTP 200 para ambos modelos.
- Suite: 13 pruebas aprobadas, 0 omitidas. Se registraron advertencias porque el entorno de preparación tenía scikit-learn 1.8.0, mientras que los modelos fueron serializados con 1.9.0. El despliegue fija 1.9.0 y se recomienda repetir allí la suite.
- `EVIDENCIA_API_LOCAL.json` conserva respuestas del cliente de pruebas; `EVIDENCIA_API_HTTP_LOCAL.json` conserva las respuestas obtenidas al levantar Uvicorn y enviar solicitudes HTTP a `127.0.0.1`.
- `EVIDENCIA_PYTEST.txt` resume la ejecución de pytest.
- `ejemplos/seguridad_prueba_muestra.csv` y `ejemplos/aplicaciones_prueba_muestra.csv` son muestras de 50 filas tomadas de los conjuntos de prueba reales.
- La UI Streamlit no se ejecutó en este entorno porque Streamlit no estaba instalado. No se generaron capturas de UI ni URLs públicas.
