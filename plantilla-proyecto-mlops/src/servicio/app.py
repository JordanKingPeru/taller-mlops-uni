"""API de predicción de churn — FastAPI + Pydantic.

Carga el modelo desde models/modelo.joblib al arrancar.
¿Tienes MLflow Registry en tu empresa? Reemplaza la carga por:
    mlflow.sklearn.load_model("models:/tu-modelo@champion")
"""
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import RAIZ, cargar_params

estado: dict = {"modelo": None, "params": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    estado["params"] = cargar_params()
    estado["modelo"] = joblib.load(RAIZ / "models" / "modelo.joblib")
    yield
    estado["modelo"] = None


app = FastAPI(title="API de predicción de churn", version="1.0", lifespan=lifespan)


class Cliente(BaseModel):
    edad: float | None = Field(None, ge=18, le=100)
    departamento: str
    plan: str
    tipo_contrato: str
    meses_antiguedad: int = Field(..., ge=1, le=200)
    cargo_mensual_soles: float = Field(..., ge=0)
    gb_datos_mes: float | None = Field(None, ge=0)
    minutos_llamadas_mes: int = Field(..., ge=0)
    lineas_adicionales: int = Field(0, ge=0, le=10)
    tickets_soporte_6m: int = Field(0, ge=0)
    caidas_servicio_mes: int = Field(0, ge=0)
    dias_ultimo_pago_vencido: int = Field(0, ge=0)
    factura_electronica: int = Field(..., ge=0, le=1)


class Prediccion(BaseModel):
    probabilidad_churn: float
    riesgo: str


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": estado["modelo"] is not None}


@app.post("/predict", response_model=Prediccion)
def predict(cliente: Cliente):
    if estado["modelo"] is None:
        raise HTTPException(503, "Modelo no cargado")
    umbral = estado["params"]["umbral_decision"]
    df = pd.DataFrame([cliente.model_dump()])
    proba = float(estado["modelo"].predict_proba(df)[0, 1])
    riesgo = "alto" if proba >= umbral else ("medio" if proba >= 0.20 else "bajo")
    return Prediccion(probabilidad_churn=round(proba, 4), riesgo=riesgo)


@app.post("/predict_lote", response_model=list[Prediccion])
def predict_lote(clientes: list[Cliente]):
    if estado["modelo"] is None:
        raise HTTPException(503, "Modelo no cargado")
    umbral = estado["params"]["umbral_decision"]
    df = pd.DataFrame([c.model_dump() for c in clientes])
    probas = estado["modelo"].predict_proba(df)[:, 1]
    out = [
        Prediccion(
            probabilidad_churn=round(float(p), 4),
            riesgo="alto" if p >= umbral else ("medio" if p >= 0.20 else "bajo"),
        )
        for p in probas
    ]
    return sorted(out, key=lambda x: -x.probabilidad_churn)
