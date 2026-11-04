#!/usr/bin/env python3
"""Interfaz Streamlit para los modelos de seguridad y aplicaciones."""
from __future__ import annotations

import io
import pandas as pd
import streamlit as st
from inference import MODEL_CONFIG, InputValidationError, load_model_bundle, predict_frame

st.set_page_config(page_title="Perfilador de tráfico de red", page_icon="📡", layout="wide")
st.title("Perfilador de tráfico de red")
st.caption(
    "Clasificación supervisada de flujos mediante características estadísticas. "
    "La aplicación no inspecciona el contenido de las comunicaciones."
)

st.warning(
    "Advertencia: los modelos se entrenaron con conjuntos de referencia (benchmarks): "
    "CICIDS2017 para clasificación de seguridad e ISCXVPN2016 para clasificación de aplicaciones. "
    "Los resultados no garantizan el mismo desempeño en redes reales o tráfico contemporáneo."
)

model_key = st.selectbox(
    "Modelo de clasificación",
    options=list(MODEL_CONFIG),
    format_func=lambda key: MODEL_CONFIG[key]["title"],
)

@st.cache_resource(show_spinner="Cargando modelo...")
def cached_bundle(key: str):
    return load_model_bundle(key)

try:
    bundle = cached_bundle(model_key)
except Exception as exc:
    st.error(f"No se pudo cargar el modelo: {exc}")
    st.info(
        f"Coloca `modelo_entrenado.joblib`, `caracteristicas.json` y `etiquetas.json` "
        f"en `{MODEL_CONFIG[model_key]['directory']}/`."
    )
    st.stop()

with st.expander("Detalles del modelo", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Características esperadas", len(bundle.features))
    col2.metric("Clases", len(bundle.labels))
    col3.metric("Dataset de entrenamiento", bundle.dataset)
    st.write("**Columnas requeridas, en el orden del modelo:**")
    st.code(", ".join(bundle.features), language="text")
    st.write("**Etiquetas:**", ", ".join(bundle.labels))

upload = st.file_uploader("Carga un CSV de flujos", type=["csv"])
if upload is not None:
    try:
        raw = pd.read_csv(upload, low_memory=False)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        st.stop()

    st.subheader("Validación del archivo")
    st.write(f"Registros: **{len(raw):,}** · columnas recibidas: **{len(raw.columns)}**")
    try:
        result, warnings = predict_frame(raw, bundle)
    except InputValidationError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    for warning in warnings:
        st.warning(warning)

    counts = result["prediccion"].value_counts().reindex(bundle.labels, fill_value=0)
    summary = counts.rename_axis("Clase").reset_index(name="Flujos")
    left, right = st.columns([1, 1.5])
    with left:
        st.metric("Flujos clasificados", f"{len(result):,}")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with right:
        st.bar_chart(summary.set_index("Clase"), y="Flujos", use_container_width=True)

    st.subheader("Predicciones por flujo")
    st.dataframe(result.head(1000), use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar resultados CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name=f"predicciones_{model_key}.csv",
        mime="text/csv",
    )
    st.caption(
        "La predicción es una estimación estadística. No constituye por sí sola evidencia "
        "de una intrusión ni identifica de forma determinista una aplicación."
    )
