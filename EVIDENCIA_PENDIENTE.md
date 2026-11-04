# Registro de evidencia para la entrega

Este documento separa las verificaciones ya ejecutadas localmente de las evidencias que solo pueden obtenerse al ejecutar la interfaz y publicar los Spaces.

## Pruebas locales completadas

- [x] Modelos reales copiados a `salidas_seguridad_sin_puerto/` y `salidas_aplicaciones/`.
- [x] Seguridad: modelo Random Forest cargado con 68 características y 2 etiquetas.
- [x] Aplicaciones: modelo HistGradientBoosting cargado con 23 características y 7 etiquetas.
- [x] Se reconstruyeron las filas de características de prueba usando los datasets procesados, la limpieza, las particiones y la semilla originales.
- [x] Las etiquetas reales reconstruidas coinciden, en orden, con `predicciones_prueba.csv` para ambos experimentos.
- [x] Paridad de inferencia: 5.957/5.957 predicciones de seguridad coinciden con el archivo exportado durante el entrenamiento.
- [x] Paridad de inferencia: 2.940/2.940 predicciones de aplicaciones coinciden con el archivo exportado durante el entrenamiento.
- [x] `GET /health` local: HTTP 200, estado `ok`, ambos modelos disponibles.
- [x] `POST /predict` local: HTTP 200 para ambos modelos, con cinco filas reales de prueba por solicitud.
- [x] Servidor Uvicorn local iniciado; solicitudes HTTP a `127.0.0.1:8765/health` y `POST /predict` respondieron correctamente.
- [x] Suite pytest: 13 aprobadas, 0 omitidas, 30 advertencias de versión (1.8.0 frente a 1.9.0).

## Pruebas de UI pendientes

- [ ] Ejecutar Streamlit en un entorno con las dependencias instaladas.
- [ ] Capturar la UI seleccionando el modelo de seguridad y mostrando su esquema de 68 características.
- [ ] Capturar predicciones, conteos y gráfico del modelo de seguridad.
- [ ] Capturar la UI seleccionando el modelo de aplicaciones y mostrando su esquema de 23 características.
- [ ] Capturar predicciones, conteos y gráfico del modelo de aplicaciones.
- [ ] Descargar y revisar un CSV desde la interfaz.

## Despliegue pendiente

- URL pública de UI:
- URL pública de API:
- Commit desplegado:
- Fecha y hora de verificación:

## Evidencias que deben guardarse después del despliegue

1. Capturas de la UI con ambos modelos y archivos de prueba.
2. CSV descargado desde la UI.
3. Salida íntegra de `pytest -q` ejecutado en el entorno final.
4. Respuesta íntegra de `GET /health` en la URL pública.
5. Respuesta íntegra de `POST /predict` para cada modelo en la URL pública.
6. Registro de que ambas URLs públicas responden y corresponden al commit informado.

## Advertencia de compatibilidad

Los artefactos serializados reportan scikit-learn 1.9.0. El entorno de verificación disponible tenía scikit-learn 1.8.0 y emitió `InconsistentVersionWarning`, aunque las predicciones coincidieron exactamente. Los archivos `requirements.txt` se fijaron en `scikit-learn==1.9.0` para que el despliegue utilice la versión de entrenamiento. Conviene repetir la suite dentro del contenedor con esa versión exacta.

## Límites de la evidencia

No se ha desplegado el proyecto en una cuenta de Hugging Face ni se han generado URLs públicas. Tampoco se han capturado pantallas de la UI. Las pruebas HTTP se realizaron contra un servidor Uvicorn local en `127.0.0.1:8765`, además de las pruebas automatizadas con FastAPI TestClient; no son solicitudes a un servicio público. La figura `figuras/agregacion_trafico.png` es un gráfico generado a partir de las predicciones reales de prueba, no una captura de la interfaz.
