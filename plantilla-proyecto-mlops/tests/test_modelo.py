"""Tests del entrenamiento: el pipeline entrena, predice sobre datos crudos y supera el mínimo."""
import pandas as pd

from src.entrenar import entrenar


def test_entrenamiento_completo(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    resultado = entrenar("data/muestra.csv")
    assert resultado["metricas_test"]["roc_auc"] >= 0.5  # el gate real está en params.yaml


def test_modelo_predice_con_nulos_y_categoria_nueva():
    """El pipeline debe tolerar lo que producción SÍ va a mandar."""
    import joblib

    from src.config import RAIZ

    entrenar("data/muestra.csv")
    modelo = joblib.load(RAIZ / "models" / "modelo.joblib")
    cliente = pd.DataFrame([{
        "edad": None,                      # nulo
        "departamento": "Madre de Dios",   # categoría no vista en entrenamiento
        "plan": "Prepago",
        "tipo_contrato": "Mensual",
        "meses_antiguedad": 2,
        "cargo_mensual_soles": 30.0,
        "gb_datos_mes": None,              # nulo
        "minutos_llamadas_mes": 100,
        "lineas_adicionales": 0,
        "tickets_soporte_6m": 3,
        "caidas_servicio_mes": 2,
        "dias_ultimo_pago_vencido": 10,
        "factura_electronica": 0,
    }])
    proba = modelo.predict_proba(cliente)[0, 1]
    assert 0.0 <= proba <= 1.0
