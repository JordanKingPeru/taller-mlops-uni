"""Entrenamiento del modelo: reproducible, con quality gate y tracking opcional.

Uso:  python -m src.entrenar  [ruta_datos_opcional]

Salidas:
  - models/modelo.joblib   (el pipeline completo: preprocesamiento + clasificador)
  - metrics.json           (métricas en test + metadatos)
  - run en MLflow si MLFLOW_TRACKING_URI está definido (opcional)
"""
import json
import os
import platform
import sys
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RAIZ, cargar_params
from src.datos import cargar


def construir_pipeline(params: dict) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("esc", StandardScaler()),
        ]), params["data"]["features_num"]),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), params["data"]["features_cat"]),
    ])
    clf = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        class_weight=params["model"]["class_weight"],
        random_state=params["random_state"],
        n_jobs=-1,
    )
    return Pipeline([("preproc", pre), ("clf", clf)])


def entrenar(path_datos: str | None = None) -> dict:
    params = cargar_params()
    df = cargar(path_datos)

    features = params["data"]["features_num"] + params["data"]["features_cat"]
    X, y = df[features], df[params["data"]["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["data"]["test_size"],
        random_state=params["random_state"], stratify=y,
    )

    modelo = construir_pipeline(params)
    modelo.fit(X_train, y_train)

    proba = modelo.predict_proba(X_test)[:, 1]
    pred = (proba >= params["umbral_decision"]).astype(int)
    metricas = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
    }

    # --- Quality gate: si el modelo no supera el mínimo, este script FALLA (y CI también) ---
    minimo = params["minimo_calidad"]["roc_auc"]
    if metricas["roc_auc"] < minimo:
        raise SystemExit(f"❌ Quality gate: roc_auc {metricas['roc_auc']} < mínimo {minimo}")

    # --- Persistencia ---
    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump(modelo, RAIZ / "models" / "modelo.joblib")
    salida = {
        "metricas_test": metricas,
        "fecha": datetime.now(timezone.utc).isoformat(),
        "filas_train": len(X_train),
        "params": params,
        "versiones": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    with open(RAIZ / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    # --- Tracking opcional (si hay servidor MLflow configurado) ---
    if os.environ.get("MLFLOW_TRACKING_URI"):
        import mlflow
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "proyecto-mlops"))
        with mlflow.start_run():
            mlflow.log_params({**params["model"], "umbral": params["umbral_decision"]})
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(modelo, name="modelo", serialization_format="cloudpickle")

    print("✅ Entrenado.", json.dumps(metricas))
    return salida


if __name__ == "__main__":
    entrenar(sys.argv[1] if len(sys.argv) > 1 else None)
