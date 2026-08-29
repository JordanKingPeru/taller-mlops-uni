"""Tests del contrato de datos."""
import pandas as pd
import pytest

from src.config import cargar_params
from src.datos import cargar, validar


def test_muestra_cumple_contrato():
    df = cargar("data/muestra.csv")
    assert len(df) >= 100
    assert df["churn"].isin([0, 1]).all()


def test_columna_faltante_falla():
    params = cargar_params()
    df = pd.read_csv("data/muestra.csv").drop(columns=["plan"])
    with pytest.raises(ValueError, match="faltan columnas"):
        validar(df, params)


def test_target_no_binario_falla():
    params = cargar_params()
    df = pd.read_csv("data/muestra.csv")
    df.loc[0, "churn"] = 7
    with pytest.raises(ValueError, match="binario"):
        validar(df, params)


def test_duplicados_se_eliminan():
    params = cargar_params()
    df = pd.read_csv("data/muestra.csv")
    df_dup = pd.concat([df, df.head(10)], ignore_index=True)
    assert len(validar(df_dup, params)) == len(validar(df, params))
