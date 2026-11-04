"""Carga, valida y ejecuta los modelos de clasificación de tráfico."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd


MODEL_CONFIG = {
    "seguridad": {
        "title": "Clasificación de seguridad",
        "directory": "salidas_seguridad_sin_puerto",
        "expected_features": 68,
        "expected_classes": 2,
        "dataset": "CICIDS2017 GeneratedLabelledFlows",
    },
    "aplicaciones": {
        "title": "Clasificación de aplicaciones/servicios",
        "directory": "salidas_aplicaciones",
        "expected_features": 23,
        "expected_classes": 7,
        "dataset": "ISCXVPN2016 Scenario B, 60s-AllinOne",
    },
}


class InputValidationError(ValueError):
    """Error de estructura o contenido en el CSV de entrada."""


@dataclass
class ModelBundle:
    key: str
    title: str
    dataset: str
    model: Any
    features: list[str]
    labels: list[str]
    model_path: Path


@dataclass
class ValidationReport:
    matrix: pd.DataFrame
    warnings: list[str]
    missing_values: dict[str, int]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_bundle(key: str, root: Path | str | None = None) -> ModelBundle:
    if key not in MODEL_CONFIG:
        raise ValueError(f"Modelo desconocido: {key!r}. Opciones: {sorted(MODEL_CONFIG)}")
    root_path = Path(root) if root is not None else Path(__file__).resolve().parent
    config = MODEL_CONFIG[key]
    output_dir = root_path / config["directory"]
    model_path = output_dir / "modelo_entrenado.joblib"
    features_path = output_dir / "caracteristicas.json"
    labels_path = output_dir / "etiquetas.json"

    absent = [str(path.relative_to(root_path)) for path in (model_path, features_path, labels_path) if not path.is_file()]
    if absent:
        raise FileNotFoundError(
            f"Faltan artefactos del modelo {key}: {', '.join(absent)}. "
            "Copia los artefactos generados por train_validate.py a la carpeta indicada."
        )

    features = _read_json(features_path)
    labels = _read_json(labels_path)
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ValueError(f"{features_path} debe contener una lista JSON de nombres de características.")
    if len(features) != config["expected_features"]:
        raise ValueError(
            f"{key}: se esperaban {config['expected_features']} características, "
            f"pero caracteristicas.json contiene {len(features)}."
        )
    if len(set(features)) != len(features):
        raise ValueError(f"{features_path} contiene nombres de características duplicados.")
    if not isinstance(labels, list) or len(labels) != config["expected_classes"]:
        raise ValueError(
            f"{key}: se esperaban {config['expected_classes']} etiquetas, "
            f"pero etiquetas.json contiene {len(labels) if isinstance(labels, list) else 'un formato inválido'}."
        )
    labels = [str(label) for label in labels]
    expected_label_sets = {
        "seguridad": {"Attack", "Benign"},
        "aplicaciones": {"Browsing", "Chat", "FileTransfer", "Mail", "P2P", "Streaming", "VoIP"},
    }
    if set(labels) != expected_label_sets[key]:
        raise ValueError(
            f"{key}: las etiquetas de etiquetas.json no coinciden con el esquema esperado: {labels}"
        )

    model = joblib.load(model_path)
    return ModelBundle(
        key=key,
        title=config["title"],
        dataset=config["dataset"],
        model=model,
        features=features,
        labels=[str(label) for label in labels],
        model_path=model_path,
    )


def validate_input_frame(frame: pd.DataFrame, bundle: ModelBundle) -> ValidationReport:
    if frame.empty:
        raise InputValidationError("El CSV no contiene registros.")
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise InputValidationError(f"El CSV contiene nombres de columnas duplicados: {duplicated}")

    # Se normalizan espacios y BOM, pero no se renombran variables semánticamente distintas.
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lstrip("\ufeff") for column in normalized.columns]
    if normalized.columns.duplicated().any():
        duplicated = normalized.columns[normalized.columns.duplicated()].astype(str).tolist()
        raise InputValidationError(f"Hay columnas duplicadas después de normalizar encabezados: {duplicated}")

    missing = [feature for feature in bundle.features if feature not in normalized.columns]
    if missing:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise InputValidationError(
            f"Faltan {len(missing)} características obligatorias: {preview}{suffix}"
        )

    matrix = pd.DataFrame(index=normalized.index)
    non_numeric: dict[str, int] = {}
    infinite: dict[str, int] = {}
    missing_values: dict[str, int] = {}

    for feature in bundle.features:
        original = normalized[feature]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & original.astype(str).str.strip().ne("") & converted.isna()
        if invalid.any():
            non_numeric[feature] = int(invalid.sum())
        numeric = converted.astype("float64")
        inf_count = int(np.isinf(numeric.to_numpy()).sum())
        if inf_count:
            infinite[feature] = inf_count
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        missing_count = int(numeric.isna().sum())
        if missing_count:
            missing_values[feature] = missing_count
        matrix[feature] = numeric

    if non_numeric:
        details = ", ".join(f"{name}: {count}" for name, count in list(non_numeric.items())[:8])
        raise InputValidationError(f"Se encontraron valores no numéricos en características requeridas ({details}).")
    if infinite:
        details = ", ".join(f"{name}: {count}" for name, count in list(infinite.items())[:8])
        raise InputValidationError(f"Se encontraron valores infinitos; corrige el CSV antes de inferir ({details}).")

    warnings: list[str] = []
    if missing_values:
        total = sum(missing_values.values())
        warnings.append(
            f"Se encontraron {total} valores vacíos/NaN en las características requeridas. "
            "El pipeline entrenado aplicará imputación mediana si contiene el paso de imputación."
        )
        has_imputer = hasattr(bundle.model, "named_steps") and any(
            "imput" in str(name).lower() for name in bundle.model.named_steps
        )
        if not has_imputer:
            raise InputValidationError(
                "El CSV contiene valores vacíos, pero el modelo cargado no expone un paso de imputación."
            )
        all_empty = [name for name in matrix.columns if matrix[name].isna().all()]
        if all_empty:
            warnings.append(
                "Estas columnas están completamente vacías en el archivo y serán imputadas: "
                + ", ".join(all_empty)
            )

    return ValidationReport(matrix=matrix, warnings=warnings, missing_values=missing_values)


def predict_frame(frame: pd.DataFrame, bundle: ModelBundle) -> tuple[pd.DataFrame, list[str]]:
    report = validate_input_frame(frame, bundle)
    try:
        encoded = np.asarray(bundle.model.predict(report.matrix))
    except Exception as exc:
        raise RuntimeError(f"Falló la inferencia con el modelo {bundle.key}: {exc}") from exc

    decoded: list[str] = []
    for value in encoded:
        # train_validate.py entrena LabelEncoder sobre etiquetas ordenadas y guarda esa lista.
        try:
            index = int(value)
        except (TypeError, ValueError):
            label = str(value)
            if label not in bundle.labels:
                raise RuntimeError(f"El modelo produjo una clase no incluida en etiquetas.json: {label!r}")
            decoded.append(label)
            continue
        if index < 0 or index >= len(bundle.labels):
            raise RuntimeError(f"El modelo produjo un índice de clase fuera de rango: {index}")
        decoded.append(bundle.labels[index])

    result = frame.copy()
    result["prediccion"] = decoded
    if hasattr(bundle.model, "predict_proba"):
        try:
            probabilities = np.asarray(bundle.model.predict_proba(report.matrix))
            if probabilities.ndim == 2 and probabilities.shape[1] == len(bundle.labels):
                result["probabilidad_maxima_modelo"] = probabilities.max(axis=1)
        except Exception:
            # La confianza es opcional; la predicción de clase sigue siendo válida.
            pass
    result["modelo"] = bundle.key
    return result, report.warnings
