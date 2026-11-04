#!/usr/bin/env python3
"""Entrena, selecciona y evalúa modelos con un CSV de características de flujo.

Selección: F1 macro sobre validación. Evaluación final: conjunto de prueba separado.
Modelos comparados: regresión logística, Random Forest, HistGradientBoosting y MLP.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


ID_PATTERN = re.compile(
    r"(^id$|flow.?id|source.?ip|destination.?ip|src.?ip|dst.?ip|timestamp|time.?stamp|date.?time|^unnamed)",
    flags=re.IGNORECASE,
)


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def group_split(y: pd.Series, groups: pd.Series, test_size: float, seed: int, all_classes_in_test: bool = True):
    labels = set(y.astype(str))
    splitter = GroupShuffleSplit(n_splits=250, test_size=test_size, random_state=seed)
    dummy_x = np.zeros((len(y), 1))
    for train_idx, test_idx in splitter.split(dummy_x, y, groups):
        train_labels = set(y.iloc[train_idx].astype(str))
        test_labels = set(y.iloc[test_idx].astype(str))
        if labels.issubset(train_labels) and (not all_classes_in_test or labels.issubset(test_labels)):
            return train_idx, test_idx
    raise ValueError(
        "No se encontró una partición por grupos que conserve todas las clases. "
        "Revisa los grupos o usa una clave de captura/sesión adecuada; no elimines el grupo "
        "solo para conseguir una métrica favorable."
    )


def make_models(seed: int) -> dict[str, Pipeline]:
    # La misma imputación y escala se ajustan solo con la partición de entrenamiento.
    def pipe(estimator: object) -> Pipeline:
        return Pipeline([
            ("imputacion", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("escalado", StandardScaler()),
            ("modelo", estimator),
        ])

    return {
        "RegresionLogistica": pipe(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        "RandomForest": pipe(RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=-1,
            random_state=seed, min_samples_leaf=1,
        )),
        "GradientBoosting": pipe(HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=seed,
        )),
        "MLP_128_64": pipe(MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
            max_iter=200, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=12, random_state=seed,
        )),
    }


def drop_identifier_columns(frame: pd.DataFrame, target: str, group_column: str | None) -> list[str]:
    drops: list[str] = []
    for column in frame.columns:
        if column == target or column == group_column:
            continue
        if ID_PATTERN.search(str(column).strip()):
            drops.append(column)
    return drops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV preparado")
    parser.add_argument("--label-column", default="Label", help="Columna objetivo")
    parser.add_argument("--group-column", default=None, help="Columna de captura/sesión para separar grupos sin fuga")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.1875, help="Fracción del conjunto no-test que se reserva para validación")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-cols", nargs="*", default=[], help="Columnas adicionales que no deben ser variables de entrada")
    args = parser.parse_args()

    if not 0.05 <= args.test_size <= 0.4:
        parser.error("--test-size debe estar entre 0.05 y 0.40")
    if not 0.05 <= args.validation_size <= 0.4:
        parser.error("--validation-size debe estar entre 0.05 y 0.40")

    df = read_csv(args.input)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    label_lookup = {c.casefold(): c for c in df.columns}
    target = label_lookup.get(args.label_column.casefold())
    if target is None:
        raise ValueError(f"No existe la etiqueta {args.label_column!r}. Columnas: {list(df.columns)}")
    df[target] = df[target].astype("string").str.strip()
    df = df[df[target].notna() & df[target].ne("")].drop_duplicates().reset_index(drop=True)
    if df[target].nunique() < 2:
        raise ValueError("La columna objetivo debe tener al menos dos clases.")
    counts = df[target].value_counts()
    if int(counts.min()) < 3:
        rare = counts[counts < 3].to_dict()
        raise ValueError(f"Cada clase necesita al menos tres registros para dividir entrenamiento/validación/prueba. Clases escasas: {rare}")

    group_col = None
    if args.group_column:
        group_lookup = {c.casefold(): c for c in df.columns}
        group_col = group_lookup.get(args.group_column.casefold())
        if group_col is None:
            raise ValueError(f"No existe la columna de grupo {args.group_column!r}.")
        groups = df[group_col].astype("string").fillna("__SIN_GRUPO__")
        outer_train_idx, test_idx = group_split(df[target], groups, args.test_size, args.seed, all_classes_in_test=True)
        outer_train = df.iloc[outer_train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        trainval_groups = outer_train[group_col].astype("string").fillna("__SIN_GRUPO__")
        inner_train_idx, val_idx = group_split(
            outer_train[target], trainval_groups, args.validation_size, args.seed + 1, all_classes_in_test=True
        )
        train_df = outer_train.iloc[inner_train_idx].reset_index(drop=True)
        val_df = outer_train.iloc[val_idx].reset_index(drop=True)
    else:
        trainval_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=args.seed, stratify=df[target]
        )
        train_df, val_df = train_test_split(
            trainval_df, test_size=args.validation_size, random_state=args.seed + 1,
            stratify=trainval_df[target],
        )
        train_df, val_df, test_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    drop_cols = set(drop_identifier_columns(df, target, group_col)) | set(args.drop_cols)
    drop_cols.discard(target)
    if group_col:
        drop_cols.add(group_col)
    # Convertibles to numeric; keep columns with at least one numeric observation and variation.
    numeric_features: list[str] = []
    converted: dict[str, pd.Series] = {}
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    for col in df.columns:
        if col == target or col in drop_cols:
            continue
        values = pd.to_numeric(combined[col], errors="coerce")
        if values.notna().mean() >= 0.50 and values.nunique(dropna=True) > 1:
            numeric_features.append(col)
            converted[col] = values.replace([np.inf, -np.inf], np.nan)
    if not numeric_features:
        raise ValueError("No se encontraron características numéricas útiles después de excluir identificadores.")

    X_all = pd.DataFrame({c: converted[c] for c in numeric_features})
    X_train = X_all.iloc[:len(train_df)].reset_index(drop=True)
    X_val = X_all.iloc[len(train_df):len(train_df) + len(val_df)].reset_index(drop=True)
    X_test = X_all.iloc[len(train_df) + len(val_df):].reset_index(drop=True)
    y_train_text = train_df[target].astype(str).reset_index(drop=True)
    y_val_text = val_df[target].astype(str).reset_index(drop=True)
    y_test_text = test_df[target].astype(str).reset_index(drop=True)
    labels = sorted(df[target].astype(str).unique().tolist())
    label_encoder = LabelEncoder().fit(labels)
    # Numeric encoding also avoids dtype-specific scoring issues in certain sklearn releases.
    y_train = pd.Series(label_encoder.transform(y_train_text))
    y_val = pd.Series(label_encoder.transform(y_val_text))
    y_test = pd.Series(label_encoder.transform(y_test_text))
    class_ids = list(range(len(labels)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    validation_models: dict[str, Pipeline] = {}
    for name, model in make_models(args.seed).items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred_val = model.predict(X_val)
        predict_seconds = time.perf_counter() - start
        validation_models[name] = model
        results.append({
            "modelo": name,
            "f1_macro_validacion": f1_score(y_val, pred_val, labels=class_ids, average="macro", zero_division=0),
            "precision_macro_validacion": precision_score(y_val, pred_val, labels=class_ids, average="macro", zero_division=0),
            "recall_macro_validacion": recall_score(y_val, pred_val, labels=class_ids, average="macro", zero_division=0),
            "accuracy_validacion": accuracy_score(y_val, pred_val),
            "entrenamiento_segundos_sobre_train": fit_seconds,
            "prediccion_ms_por_flujo_validacion": (predict_seconds / max(len(X_val), 1)) * 1000.0,
        })
    validation_results = pd.DataFrame(results).sort_values(
        ["f1_macro_validacion", "accuracy_validacion"], ascending=False
    ).reset_index(drop=True)
    validation_results.to_csv(args.output_dir / "comparacion_validacion.csv", index=False)
    best_name = str(validation_results.iloc[0]["modelo"])

    # Reentrena el modelo elegido usando train + validation; test se conserva sin tocar.
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    final_model = make_models(args.seed)[best_name]
    final_model.fit(X_trainval, y_trainval)
    start = time.perf_counter()
    pred_test = final_model.predict(X_test)
    test_predict_seconds = time.perf_counter() - start

    metrics = {
        "modelo_seleccionado_por_f1_macro_validacion": best_name,
        "registros_totales": int(len(df)),
        "registros_entrenamiento": int(len(X_train)),
        "registros_validacion": int(len(X_val)),
        "registros_prueba": int(len(X_test)),
        "numero_caracteristicas": int(len(numeric_features)),
        "numero_clases": int(len(labels)),
        "clases": labels,
        "accuracy_prueba": float(accuracy_score(y_test, pred_test)),
        "precision_macro_prueba": float(precision_score(y_test, pred_test, labels=class_ids, average="macro", zero_division=0)),
        "recall_macro_prueba": float(recall_score(y_test, pred_test, labels=class_ids, average="macro", zero_division=0)),
        "f1_macro_prueba": float(f1_score(y_test, pred_test, labels=class_ids, average="macro", zero_division=0)),
        "tiempo_prediccion_prueba_segundos": float(test_predict_seconds),
        "tiempo_prediccion_prueba_ms_por_flujo": float(test_predict_seconds / max(len(X_test), 1) * 1000.0),
        "semilla": args.seed,
        "particion": "por grupos" if group_col else "estratificada aleatoria",
    }
    (args.output_dir / "metricas_prueba.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(classification_report(
        y_test, pred_test, labels=class_ids, target_names=labels, output_dict=True, zero_division=0
    )).T.to_csv(args.output_dir / "reporte_por_clase.csv", index=True, index_label="clase")
    cm = confusion_matrix(y_test, pred_test, labels=class_ids)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(args.output_dir / "matriz_confusion.csv", index=True, index_label="real/predicho")

    # Confusion matrix figure with adaptive size and legible labels.
    fig_width = max(8, min(18, 0.65 * len(labels) + 4))
    fig_height = max(6, min(16, 0.60 * len(labels) + 3))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(cm, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title=f"Matriz de confusión en prueba: {best_name}",
        xlabel="Clase predicha", ylabel="Clase real",
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
    )
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    if len(labels) <= 15:
        threshold = cm.max() / 2 if cm.size and cm.max() else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                        color="white" if cm[i, j] > threshold else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "matriz_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Exporta exactamente las características del conjunto de prueba para pruebas de regresión
    # de la inferencia desplegada. Mantiene el mismo orden que predicciones_prueba.csv.
    X_test.to_csv(args.output_dir / "caracteristicas_prueba.csv", index=False)
    pd.DataFrame({
        "real": label_encoder.inverse_transform(y_test.to_numpy()),
        "prediccion": label_encoder.inverse_transform(np.asarray(pred_test, dtype=int)),
    }).to_csv(args.output_dir / "predicciones_prueba.csv", index=False)
    joblib.dump(final_model, args.output_dir / "modelo_entrenado.joblib")
    (args.output_dir / "caracteristicas.json").write_text(json.dumps(numeric_features, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "etiquetas.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "configuracion.json").write_text(json.dumps({
        "modelos_comparados": list(make_models(args.seed).keys()),
        "mlp": {"capas_ocultas": [128, 64], "activacion_oculta": "ReLU", "optimizador": "Adam",
                "tasa_aprendizaje": 0.001, "regularizacion_l2_alpha": 0.0001,
                "max_iteraciones": 200, "early_stopping": True, "validacion_interna": 0.15},
        "caracteristicas_utilizadas": numeric_features,
        "columnas_excluidas": sorted(drop_cols),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nArtefactos guardados en: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
