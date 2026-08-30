"""Servicio de predicción de churn — AndesTel. Taller MLOps UNI."""
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

NOMBRE_MODELO = "churn-andestel"
UMBRAL_ALERTA = 0.35  # decidido con negocio (Lab 1): priorizamos recall

estado = {"modelo": None, "version": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al arrancar: cargar el champion desde el Registry (una sola vez, no por request)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    estado["modelo"] = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
    estado["version"] = mv.version
    yield
    estado["modelo"] = None

app = FastAPI(title="API Churn AndesTel", version="1.0", lifespan=lifespan)

# --- Contrato de entrada: Pydantic valida tipos y rangos ANTES de llegar al modelo ---
class Plan(str, Enum):
    prepago = "Prepago"
    basico = "Postpago Básico"
    plus = "Postpago Plus"
    premium = "Postpago Premium"

class Contrato(str, Enum):
    mensual = "Mensual"
    anual = "Anual"
    m18 = "18 meses"

class Cliente(BaseModel):
    edad: Optional[float] = Field(None, ge=18, le=100)
    departamento: str
    plan: Plan
    tipo_contrato: Contrato
    meses_antiguedad: int = Field(..., ge=1, le=200)
    cargo_mensual_soles: float = Field(..., ge=0)
    gb_datos_mes: Optional[float] = Field(None, ge=0)
    minutos_llamadas_mes: int = Field(..., ge=0)
    lineas_adicionales: int = Field(0, ge=0, le=10)
    tickets_soporte_6m: int = Field(0, ge=0)
    caidas_servicio_mes: int = Field(0, ge=0)
    dias_ultimo_pago_vencido: int = Field(0, ge=0)
    factura_electronica: int = Field(..., ge=0, le=1)

class Prediccion(BaseModel):
    probabilidad_churn: float
    riesgo: str
    accion_sugerida: str
    version_modelo: str

@app.get("/health")
def health():
    return {"status": "ok", "modelo": NOMBRE_MODELO, "version": estado["version"]}

@app.post("/predict", response_model=Prediccion)
def predict(cliente: Cliente):
    if estado["modelo"] is None:
        raise HTTPException(503, "Modelo no cargado")
    df = pd.DataFrame([cliente.model_dump()])
    # Pydantic entrega Enums; el pipeline espera strings
    df["plan"] = df["plan"].map(lambda x: x.value if hasattr(x, "value") else x)
    df["tipo_contrato"] = df["tipo_contrato"].map(lambda x: x.value if hasattr(x, "value") else x)
    proba = float(estado["modelo"].predict_proba(df)[0, 1])
    riesgo = "alto" if proba >= UMBRAL_ALERTA else ("medio" if proba >= 0.20 else "bajo")
    accion = {
        "alto": "Derivar a retención con oferta personalizada",
        "medio": "Incluir en campaña de fidelización del mes",
        "bajo": "Sin acción",
    }[riesgo]
    return Prediccion(probabilidad_churn=round(proba, 4), riesgo=riesgo,
                      accion_sugerida=accion, version_modelo=str(estado["version"]))
