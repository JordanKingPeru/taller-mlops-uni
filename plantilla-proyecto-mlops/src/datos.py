"""Carga y validación de datos: el contrato de datos del proyecto.

Si los datos no cumplen el contrato, el entrenamiento DEBE fallar ruidosamente.
Adapta `validar` a tu propio dataset cuando reutilices la plantilla.
"""
import pandas as pd

from src.config import cargar_params


def cargar(path: str | None = None) -> pd.DataFrame:
    params = cargar_params()
    df = pd.read_csv(path or params["data"]["path"])
    return validar(df, params)


def validar(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    target = params["data"]["target"]
    columnas = set(params["data"]["features_num"]) | set(params["data"]["features_cat"]) | {target}

    errores = []
    if faltan := columnas - set(df.columns):
        errores.append(f"faltan columnas: {sorted(faltan)}")
    if target in df.columns and not df[target].dropna().isin([0, 1]).all():
        errores.append(f"'{target}' debe ser binario (0/1)")
    if "cargo_mensual_soles" in df.columns and (df["cargo_mensual_soles"] < 0).any():
        errores.append("cargos negativos")
    if len(df) < 100:
        errores.append(f"muy pocas filas para entrenar: {len(df)}")
    if errores:
        raise ValueError("Contrato de datos violado: " + "; ".join(errores))

    if "id_cliente" in df.columns:
        df = df.drop_duplicates(subset="id_cliente", keep="first")
    return df.reset_index(drop=True)
