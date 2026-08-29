"""Evalúa el modelo guardado contra un dataset (por defecto, el de params.yaml).

Uso:  python -m src.evaluar [ruta_datos]
Útil para: validar el modelo contra un lote nuevo antes de promoverlo.
"""
import json
import sys

import joblib
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.config import RAIZ, cargar_params
from src.datos import cargar


def evaluar(path_datos: str | None = None) -> dict:
    params = cargar_params()
    modelo = joblib.load(RAIZ / "models" / "modelo.joblib")
    df = cargar(path_datos)

    features = params["data"]["features_num"] + params["data"]["features_cat"]
    proba = modelo.predict_proba(df[features])[:, 1]
    pred = (proba >= params["umbral_decision"]).astype(int)
    y = df[params["data"]["target"]]

    resultado = {
        "n_filas": len(df),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "tasa_alertas": round(float(pred.mean()), 4),
    }
    print(json.dumps(resultado, indent=2))
    return resultado


if __name__ == "__main__":
    evaluar(sys.argv[1] if len(sys.argv) > 1 else None)
