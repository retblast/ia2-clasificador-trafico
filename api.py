"""API HTTP de inferencia compartiendo la lógica de inference.py."""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from inference import MODEL_CONFIG, InputValidationError, load_model_bundle, predict_frame

app = FastAPI(
    title="API de clasificación de tráfico de red",
    version="1.0.0",
    description="Inferencia sobre flujos usando modelos entrenados con CICIDS2017 e ISCXVPN2016.",
)


@lru_cache(maxsize=2)
def get_bundle(key: str):
    return load_model_bundle(key)


@app.get("/health")
def health():
    model_status = {}
    for key in MODEL_CONFIG:
        try:
            bundle = get_bundle(key)
            model_status[key] = {
                "available": True,
                "features": len(bundle.features),
                "classes": bundle.labels,
            }
        except Exception as exc:
            model_status[key] = {"available": False, "error": str(exc)}
    all_available = all(item["available"] for item in model_status.values())
    return {
        "status": "ok" if all_available else "degraded",
        "models": model_status,
    }


@app.post("/predict")
async def predict(
    model: str = Form(..., description="seguridad o aplicaciones"),
    file: UploadFile = File(..., description="CSV con las características del modelo seleccionado"),
):
    if model not in MODEL_CONFIG:
        raise HTTPException(status_code=422, detail=f"Modelo inválido. Opciones: {list(MODEL_CONFIG)}")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Debe enviarse un archivo .csv")
    try:
        content = await file.read()
        frame = pd.read_csv(BytesIO(content), low_memory=False)
        bundle = get_bundle(model)
        result, warnings = predict_frame(frame, bundle)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo procesar el CSV: {exc}") from exc

    return {
        "model": model,
        "dataset": bundle.dataset,
        "rows": len(result),
        "warnings": warnings,
        "predictions": result.to_dict(orient="records"),
    }
