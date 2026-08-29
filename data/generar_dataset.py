"""
Generador del dataset del taller: churn de clientes de una telco peruana ficticia ("AndesTel").
Datos sintéticos pero con relaciones realistas entre variables y el target.
Se generan dos archivos:
  - churn_telco_peru.csv        : dataset de entrenamiento (periodo 2025)
  - churn_telco_peru_nuevos.csv : lote "de producción" (periodo 2026) con drift inyectado,
                                  usado en el lab de monitoreo (sin columna target aparte del oculto)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DEPARTAMENTOS = ["Lima", "Arequipa", "La Libertad", "Piura", "Cusco", "Junín", "Lambayeque", "Áncash"]
DEP_P = [0.42, 0.10, 0.09, 0.08, 0.08, 0.08, 0.08, 0.07]
PLANES = ["Prepago", "Postpago Básico", "Postpago Plus", "Postpago Premium"]
PLAN_P = [0.38, 0.30, 0.22, 0.10]
CONTRATOS = ["Mensual", "Anual", "18 meses"]


def generar(n: int, drift: bool = False, seed_offset: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(42 + seed_offset)

    plan = rng.choice(PLANES, n, p=PLAN_P if not drift else [0.50, 0.28, 0.15, 0.07])
    plan_idx = pd.Series(plan).map({p: i for i, p in enumerate(PLANES)}).values

    meses_antiguedad = np.clip(rng.exponential(24, n).astype(int) + 1, 1, 96)
    if drift:
        # En producción llegan más clientes nuevos (campaña de captación agresiva)
        meses_antiguedad = np.clip(rng.exponential(12, n).astype(int) + 1, 1, 96)

    base_cargo = np.array([25, 45, 70, 110])[plan_idx]
    cargo_mensual = np.round(base_cargo * rng.normal(1.0, 0.15, n), 2)
    if drift:
        cargo_mensual = np.round(cargo_mensual * 1.18, 2)  # subida de tarifas

    contrato = np.where(
        plan_idx == 0, "Mensual",
        rng.choice(CONTRATOS, n, p=[0.45, 0.35, 0.20])
    )

    gb_datos_mes = np.round(np.clip(rng.gamma(2.5, 4 + 2 * plan_idx), 0.1, 120), 1)
    minutos_llamadas = np.clip(rng.normal(300 + 80 * plan_idx, 120, n), 0, None).astype(int)
    tickets_soporte_6m = rng.poisson(0.8, n)
    caidas_servicio_mes = rng.poisson(1.2 if not drift else 2.1, n)  # red degradada en 2026
    dias_ultimo_pago_vencido = np.clip(rng.exponential(4, n).astype(int), 0, 60)
    factura_electronica = rng.choice([0, 1], n, p=[0.35, 0.65])
    lineas_adicionales = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.28, 0.12, 0.05])
    edad = np.clip(rng.normal(38, 13, n).astype(int), 18, 85)
    departamento = rng.choice(DEPARTAMENTOS, n, p=DEP_P)

    # ----- Probabilidad de churn (relaciones realistas) -----
    logit = (
        -1.9
        - 0.030 * meses_antiguedad
        + 0.45 * tickets_soporte_6m
        + 0.28 * caidas_servicio_mes
        + 0.035 * dias_ultimo_pago_vencido
        + 0.012 * (cargo_mensual - base_cargo)          # pagar más de lo esperado molesta
        - 0.35 * (contrato == "Anual")
        - 0.55 * (contrato == "18 meses")
        - 0.25 * lineas_adicionales
        - 0.20 * factura_electronica
        + 0.30 * (plan == "Prepago")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "id_cliente": [f"CL{100000 + i + seed_offset * 100000}" for i in range(n)],
        "edad": edad,
        "departamento": departamento,
        "plan": plan,
        "tipo_contrato": contrato,
        "meses_antiguedad": meses_antiguedad,
        "cargo_mensual_soles": cargo_mensual,
        "gb_datos_mes": gb_datos_mes,
        "minutos_llamadas_mes": minutos_llamadas,
        "lineas_adicionales": lineas_adicionales,
        "tickets_soporte_6m": tickets_soporte_6m,
        "caidas_servicio_mes": caidas_servicio_mes,
        "dias_ultimo_pago_vencido": dias_ultimo_pago_vencido,
        "factura_electronica": factura_electronica,
        "churn": churn,
    })

    # Imperfecciones realistas para el lab de limpieza/reproducibilidad
    idx_nan = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[idx_nan, "gb_datos_mes"] = np.nan
    idx_nan2 = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[idx_nan2, "edad"] = np.nan
    return df


if __name__ == "__main__":
    train = generar(7500)
    prod = generar(1500, drift=True, seed_offset=7)
    # Algunos duplicados a propósito en train
    train = pd.concat([train, train.sample(60, random_state=1)], ignore_index=True)
    train = train.sample(frac=1, random_state=2).reset_index(drop=True)

    train.to_csv("churn_telco_peru.csv", index=False)
    prod.to_csv("churn_telco_peru_nuevos.csv", index=False)
    print("train:", train.shape, "| churn rate:", round(train.churn.mean(), 3))
    print("prod :", prod.shape, "| churn rate:", round(prod.churn.mean(), 3))
