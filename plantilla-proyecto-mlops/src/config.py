"""Carga de configuración: un solo punto de verdad (params.yaml)."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


def cargar_params(path: str | Path = RAIZ / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
