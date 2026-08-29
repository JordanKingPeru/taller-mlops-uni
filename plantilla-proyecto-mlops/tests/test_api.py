"""Tests de la API: contrato de entrada/salida del servicio."""
from fastapi.testclient import TestClient

from src.entrenar import entrenar
from src.servicio.app import app

CLIENTE_OK = {
    "edad": 30, "departamento": "Lima", "plan": "Prepago", "tipo_contrato": "Mensual",
    "meses_antiguedad": 3, "cargo_mensual_soles": 29.9, "gb_datos_mes": 10.0,
    "minutos_llamadas_mes": 150, "lineas_adicionales": 0, "tickets_soporte_6m": 2,
    "caidas_servicio_mes": 1, "dias_ultimo_pago_vencido": 5, "factura_electronica": 1,
}


def setup_module(module):
    entrenar("data/muestra.csv")  # asegura que exista models/modelo.joblib


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["modelo_cargado"] is True


def test_predict_ok():
    with TestClient(app) as client:
        r = client.post("/predict", json=CLIENTE_OK)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["probabilidad_churn"] <= 1.0
        assert body["riesgo"] in {"bajo", "medio", "alto"}


def test_entrada_invalida_es_422():
    with TestClient(app) as client:
        r = client.post("/predict", json=dict(CLIENTE_OK, edad=15))
        assert r.status_code == 422


def test_campo_faltante_es_422():
    payload = {k: v for k, v in CLIENTE_OK.items() if k != "factura_electronica"}
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


def test_predict_lote_ordenado():
    with TestClient(app) as client:
        riesgoso = dict(CLIENTE_OK, tickets_soporte_6m=6, caidas_servicio_mes=5,
                        dias_ultimo_pago_vencido=30)
        r = client.post("/predict_lote", json=[CLIENTE_OK, riesgoso])
        assert r.status_code == 200
        probas = [p["probabilidad_churn"] for p in r.json()]
        assert probas == sorted(probas, reverse=True)
