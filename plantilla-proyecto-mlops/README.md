# 🚀 Plantilla de Proyecto MLOps

> Plantilla del taller **"MLOps en la práctica: del notebook a producción"** — UNI, Especialización en Ciencia de Datos.
> Úsala como punto de partida para tus proyectos reales: clona, renombra y reemplaza el caso de churn por tu problema.

Estructura mínima pero completa de un proyecto de ML listo para producción: código modular, tests, entrenamiento con tracking, API de serving, Docker y CI en GitHub Actions.

## Estructura

```
plantilla-proyecto-mlops/
├── README.md                  ← estás aquí
├── requirements.txt           ← dependencias pinneadas
├── params.yaml                ← TODA la configuración del proyecto (versionada)
├── Makefile                   ← comandos estándar: make train, make test, make serve...
├── Dockerfile                 ← imagen del servicio de predicción
├── .github/workflows/ci.yml   ← CI: lint + tests + entrenamiento smoke + build Docker
├── data/
│   └── muestra.csv            ← muestra pequeña para tests y CI (los datos reales NO van a Git)
├── src/
│   ├── config.py              ← carga params.yaml (un solo punto de verdad)
│   ├── datos.py               ← carga y validación de datos
│   ├── entrenar.py            ← entrena el pipeline y guarda modelo + métricas (+ MLflow opcional)
│   ├── evaluar.py             ← evalúa un modelo guardado contra un dataset
│   └── servicio/app.py        ← API FastAPI de predicción
└── tests/
    ├── test_datos.py          ← el contrato de datos se cumple
    ├── test_modelo.py         ← el pipeline entrena y supera un mínimo de calidad
    └── test_api.py            ← la API responde y valida entradas
```

## Uso rápido

```bash
# 1. Crear entorno e instalar
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Poner tus datos (o usa la muestra para probar)
cp ../churn_telco_peru.csv data/  # o ajusta data.path en params.yaml

# 3. Entrenar (guarda models/modelo.joblib + metrics.json, y registra en MLflow si está configurado)
make train

# 4. Tests
make test

# 5. Levantar la API
make serve   # → http://localhost:8000/docs

# 6. Docker
make docker-build && make docker-run
```

## 🧪 Lab 4 del taller: CI/CD con GitHub Actions

1. **Crea un repo en tu GitHub** (público) llamado `mi-proyecto-mlops` y sube esta carpeta:
   ```bash
   cd plantilla-proyecto-mlops
   git init && git add . && git commit -m "Proyecto MLOps inicial"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/mi-proyecto-mlops.git
   git push -u origin main
   ```
2. Entra a la pestaña **Actions** de tu repo: verás el pipeline `CI` ejecutándose solo. Se dispara en cada push.
3. Lee `.github/workflows/ci.yml` y ubica los 4 jobs: `lint`, `tests`, `train-smoke`, `docker`.
4. **Rompe algo a propósito** (p. ej. en `src/datos.py` cambia un nombre de columna), haz push y mira cómo CI lo atrapa **antes** de que llegue a producción. Arréglalo y push de nuevo.
5. Reto: agrega un job que publique `metrics.json` como *artifact* del workflow (pista: `actions/upload-artifact`).

## Cómo adaptarla a TU proyecto (el lunes, en tu trabajo)

1. `params.yaml` → cambia dataset, target, features y modelo.
2. `src/datos.py` → escribe las validaciones de TU contrato de datos.
3. `src/entrenar.py` → casi no cambia (esa es la gracia del pipeline).
4. `src/servicio/app.py` → ajusta el esquema Pydantic a tus features.
5. `tests/` → ajusta los tests al nuevo contrato. **No los borres: son tu red de seguridad.**
6. Configura `MLFLOW_TRACKING_URI` hacia el servidor de tu equipo para tracking compartido.

## Decisiones de diseño (y por qué)

- **El preprocesamiento vive dentro del Pipeline de sklearn** → imposible el *training-serving skew*.
- **La API carga un artefacto local (`models/modelo.joblib`)** → simple y sin dependencias externas al arrancar. Si tienes MLflow Registry en tu empresa, cambia una línea en `app.py` para cargar `models:/tu-modelo@champion`.
- **`params.yaml` versionado en Git** → cada commit define un experimento reproducible.
- **CI entrena un modelo "smoke" con la muestra** → si el entrenamiento se rompe, te enteras en el PR, no el día del re-entrenamiento de urgencia.
- **Los datos reales nunca van a Git** (`.gitignore`) → para versionar datos usa DVC o un data lake; aquí va solo la muestra sintética.

---
*Material del taller MLOps UNI 2026 — libre para uso y adaptación.*
