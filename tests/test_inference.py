from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from inference import (
    InputValidationError,
    ModelBundle,
    load_model_bundle,
    predict_frame,
    validate_input_frame,
)

ROOT = Path(__file__).resolve().parents[1]


class DummyModel:
    named_steps = {"imputacion": object()}


def dummy_bundle(features=("f1", "f2")):
    return ModelBundle(
        key="dummy",
        title="Modelo de prueba",
        dataset="fixture",
        model=DummyModel(),
        features=list(features),
        labels=["A", "B"],
        model_path=ROOT / "dummy.joblib",
    )


def test_validates_and_orders_required_features():
    frame = pd.DataFrame({"extra": [7], "f2": ["2.5"], "f1": ["1.0"]})
    report = validate_input_frame(frame, dummy_bundle())
    assert list(report.matrix.columns) == ["f1", "f2"]
    assert report.matrix.iloc[0].tolist() == [1.0, 2.5]


def test_rejects_missing_columns():
    frame = pd.DataFrame({"f1": [1.0]})
    with pytest.raises(InputValidationError, match="Faltan 1 características"):
        validate_input_frame(frame, dummy_bundle())


def test_rejects_non_numeric_values():
    frame = pd.DataFrame({"f1": ["not-a-number"], "f2": [2.0]})
    with pytest.raises(InputValidationError, match="no numéricos"):
        validate_input_frame(frame, dummy_bundle())


def test_rejects_infinite_values():
    frame = pd.DataFrame({"f1": [float("inf")], "f2": [2.0]})
    with pytest.raises(InputValidationError, match="infinitos"):
        validate_input_frame(frame, dummy_bundle())


def test_reports_empty_values_for_pipeline_with_imputer():
    frame = pd.DataFrame({"f1": [None], "f2": [2.0]})
    report = validate_input_frame(frame, dummy_bundle())
    assert report.warnings
    assert report.missing_values["f1"] == 1


@pytest.mark.parametrize("model_key", ["seguridad", "aplicaciones"])
def test_real_test_predictions_match_training_export(model_key):
    output_dir = ROOT / (
        "salidas_seguridad_sin_puerto" if model_key == "seguridad" else "salidas_aplicaciones"
    )
    required = [
        output_dir / "modelo_entrenado.joblib",
        output_dir / "caracteristicas.json",
        output_dir / "etiquetas.json",
        output_dir / "caracteristicas_prueba.csv",
        output_dir / "predicciones_prueba.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        pytest.skip(
            "Prueba de paridad real pendiente; faltan artefactos: " + ", ".join(missing)
        )

    bundle = load_model_bundle(model_key, ROOT)
    test_features = pd.read_csv(output_dir / "caracteristicas_prueba.csv")
    expected = pd.read_csv(output_dir / "predicciones_prueba.csv")
    result, _ = predict_frame(test_features, bundle)

    assert len(result) == len(expected)
    assert result["prediccion"].astype(str).tolist() == expected["prediccion"].astype(str).tolist()


def test_api_health_reports_missing_models(monkeypatch):
    from fastapi.testclient import TestClient
    import api

    def unavailable(key):
        raise FileNotFoundError(f"missing {key}")

    monkeypatch.setattr(api, "get_bundle", unavailable)
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert not body["models"]["seguridad"]["available"]
    assert not body["models"]["aplicaciones"]["available"]


def test_api_rejects_unknown_model():
    from fastapi.testclient import TestClient
    import api

    response = TestClient(api.app).post(
        "/predict",
        data={"model": "unknown"},
        files={"file": ("sample.csv", b"f1,f2\\n1,2\\n", "text/csv")},
    )
    assert response.status_code == 422


def test_decodes_encoded_predictions_and_preserves_original_columns():
    class PredictingModel:
        named_steps = {"imputacion": object()}

        def predict(self, matrix):
            return [1, 0]

        def predict_proba(self, matrix):
            return [[0.1, 0.9], [0.8, 0.2]]

    bundle = ModelBundle(
        key="dummy",
        title="Modelo de prueba",
        dataset="fixture",
        model=PredictingModel(),
        features=["f1", "f2"],
        labels=["Attack", "Benign"],
        model_path=ROOT / "dummy.joblib",
    )
    source = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "origen": ["x", "y"]})
    result, warnings = predict_frame(source, bundle)
    assert result["prediccion"].tolist() == ["Benign", "Attack"]
    assert result["modelo"].tolist() == ["dummy", "dummy"]
    assert result["origen"].tolist() == ["x", "y"]
    assert result["probabilidad_maxima_modelo"].tolist() == [0.9, 0.8]
    assert warnings == []


def test_api_health_with_real_models():
    from fastapi.testclient import TestClient
    import api

    api.get_bundle.cache_clear()
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models"]["seguridad"]["features"] == 68
    assert body["models"]["aplicaciones"]["features"] == 23


@pytest.mark.parametrize(
    ("model_key", "directory"),
    [
        ("seguridad", "salidas_seguridad_sin_puerto"),
        ("aplicaciones", "salidas_aplicaciones"),
    ],
)
def test_api_predicts_real_test_samples(model_key, directory):
    from fastapi.testclient import TestClient
    import api

    api.get_bundle.cache_clear()
    test_features = pd.read_csv(ROOT / directory / "caracteristicas_prueba.csv").head(5)
    expected = pd.read_csv(ROOT / directory / "predicciones_prueba.csv").head(5)
    response = TestClient(api.app).post(
        "/predict",
        data={"model": model_key},
        files={
            "file": (
                "muestra_prueba.csv",
                test_features.to_csv(index=False).encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == model_key
    assert body["rows"] == 5
    observed = [row["prediccion"] for row in body["predictions"]]
    assert observed == expected["prediccion"].astype(str).tolist()
