# 🚀 Plantilla de Proyecto MLOps (Guía Completa para Windows y Linux)

![Python Version](<https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue>)
![OS Windows](<https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-brightgreen>)
![Docker](https://img.shields.io/badge/Docker-Opcional-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)

Plantilla del taller **"MLOps en la práctica: del notebook a producción"** — **UNI, Especialización en Ciencia de Datos**.

Esta guía está diseñada específicamente para funcionar **directamente en Windows (PowerShell / CMD)** sin necesidad de instalar Docker ni depender de una versión estricta de Python.

---

## 📋 Tabla de Contenidos

- [🎯 Visión General](#-visión-general)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [💻 Guía Paso a Paso para Windows (Sin Docker)](#-guía-paso-a-paso-para-windows-sin-docker)
  - [1. Crear y Activar el Entorno Virtual](#1-crear-y-activar-el-entorno-virtual)
  - [2. Instalar Dependencias](#2-instalar-dependencias)
  - [3. Colocar tus Datos](#3-colocar-tus-datos)
  - [4. Entrenar el Modelo Localmente](#4-entrenar-el-modelo-localmente)
  - [5. Ejecutar los Tests](#5-ejecutar-los-tests)
  - [6. Levantar la API de Predicción (FastAPI)](#6-levantar-la-api-de-predicción-fastapi)
- [⚡ Helper Script para PowerShell (`run.ps1`)](#-helper-script-para-powershell-runps1)
- [🐳 Uso con Docker (Opcional)](#-uso-con-docker-opcional)
- [🧪 CI/CD con GitHub Actions](#-cicd-con-github-actions)
- [🛠️ Solución de Problemas Frecuentes (Windows Troubleshooting)](#️-solución-de-problemas-frecuentes-windows-troubleshooting)

---

## 🎯 Visión General

El objetivo de esta plantilla es transformar un análisis de machine learning (notebook) en un servicio con **calidad de software profesional**:

- **Configuración centralizada**: `params.yaml` es el único punto de verdad.
- **Pipeline de scikit-learn**: Evita el *training-serving skew* incluyendo preprocesamiento y modelo en un único artefacto.
- **Quality Gate**: El entrenamiento falla automáticamente si el modelo no alcanza la métrica mínima deseada.
- **API en tiempo real**: Construida con **FastAPI** y validación de tipos mediante **Pydantic**.
- **Tests Automatizados**: Cobertura de contrato de datos, métricas del modelo y respuestas del endpoint REST.

---

## 📁 Estructura del Proyecto

```text
plantilla-proyecto-mlops/
├── README.md                  ← Guía principal del proyecto
├── run.ps1                    ← Script de comandos rápidos para PowerShell (sustituye Make en Windows)
├── Makefile                   ← Comandos para sistemas Unix/Linux
├── requirements.txt           ← Dependencias compatibles (Python 3.10 - 3.13)
├── params.yaml                ← Configuración global del proyecto (hiperparámetros, umbrales, columnas)
├── Dockerfile                 ← Imagen para despliegue en contenedores (opcional)
├── .github/workflows/ci.yml   ← Pipeline CI de GitHub Actions
├── data/
│   └── muestra.csv            ← Dataset de prueba/smoke test
├── models/
│   └── modelo.joblib          ← Pipeline serializado generado por entrenar.py
├── src/
│   ├── config.py              ← Módulo de lectura de params.yaml
│   ├── datos.py               ← Validación del contrato de datos
│   ├── entrenar.py            ← Pipeline de entrenamiento + persistencia + quality gate
│   ├── evaluar.py             ← Evaluación de modelos guardados
│   └── servicio/app.py        ← API REST en FastAPI
└── tests/
    ├── test_datos.py          ← Validaciones del contrato de datos
    ├── test_modelo.py         ← Validaciones del modelo y quality gate
    └── test_api.py            ← Validaciones de respuestas HTTP de FastAPI
```

---

## 💻 Guía Paso a Paso para Windows (Sin Docker)

Abre tu terminal en Windows (**PowerShell** o **CMD**) y dirígete a la carpeta `plantilla-proyecto-mlops`:

```powershell
cd plantilla-proyecto-mlops
```

### 1. Crear y Activar el Entorno Virtual

> [!IMPORTANT]
> Si en PowerShell recibes un error diciendo que la ejecución de scripts está deshabilitada, ejecuta este comando una sola vez en tu sesión de PowerShell antes de activar:
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

**En PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En CMD (Símbolo del sistema):**

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

*(Sabrás que está activo porque aparecerá `(.venv)` al inicio de tu línea de comandos).*

---

### 2. Instalar Dependencias

Actualiza `pip` e instala todas las librerías necesarias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Colocar tus Datos

Puedes trabajar directamente con la muestra sintética incluida en `data/muestra.csv` o copiar el dataset completo del taller:

**En PowerShell:**

```powershell
Copy-Item ..\notebooks\churn_telco_peru.csv data\
```

**En CMD:**

```cmd
copy ..\notebooks\churn_telco_peru.csv data\
```

*Si usas otro archivo o cambias de nombre, actualiza la ruta `data.path` dentro de `params.yaml`.*

---

### 4. Entrenar el Modelo Localmente

> [!TIP]
> **¿Por qué esto resuelve las diferencias de versión de Python?**
> Al ejecutar `src.entrenar`, el pipeline genera un artefacto `models/modelo.joblib` entrenado y serializado **con la versión exacta de Python que tienes en tu máquina** (sea 3.10, 3.11, 3.12 o 3.13). Esto previene cualquier incompatibilidad de `joblib`/`pickle`.

Ejecuta el entrenamiento:

```powershell
python -m src.entrenar
```

**Resultado esperado:**

- Generación/actualización de `models/modelo.joblib`.
- Generación de `metrics.json` con los resultados en test (`roc_auc`, `f1`, `precision`, `recall`).
- Validación del **Quality Gate** (definido en `params.yaml`).

---

### 5. Ejecutar los Tests

Comprueba que el contrato de datos, la lógica del modelo y los endpoints de la API funcionen correctamente:

```powershell
python -m pytest tests/ -v
```

Deberás ver los 11 tests ejecutándose en verde `PASSED`.

---

### 6. Levantar la API de Predicción (FastAPI)

Inicia la API en servidor local usando Uvicorn:

```powershell
python -m uvicorn src.servicio.app:app --reload --host 127.0.0.1 --port 8000
```

1. **Documentación Swagger UI (Navegador):**
   Abre tu navegador e ingresa a: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
   Ahí podrás probar interactivamente las predicciones individuales (`POST /predict`) y en lote (`POST /predict_lote`).
2. **Probar desde otra ventana de PowerShell:**

   ```powershell
   $body = @{
       edad = 35
       departamento = "Lima"
       plan = "Prepago"
       tipo_contrato = "Mensual"
       meses_antiguedad = 12
       cargo_mensual_soles = 45.0
       gb_datos_mes = 15.0
       minutos_llamadas_mes = 300
       lineas_adicionales = 1
       tickets_soporte_6m = 3
       caidas_servicio_mes = 2
       dias_ultimo_pago_vencido = 10
       factura_electronica = 1
   } | ConvertTo-Json

   Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
   ```

---

## ⚡ Helper Script para PowerShell (`run.ps1`)

Para mayor comodidad en Windows (sin tener que recordar los comandos largos de Python ni requerir `make`), se incluye el script `run.ps1`:

| Comando               | Acción Equivalente                                 |
| :-------------------- | :-------------------------------------------------- |
| `.\run.ps1 train`   | `python -m src.entrenar`                          |
| `.\run.ps1 test`    | `python -m pytest tests/ -v`                      |
| `.\run.ps1 serve`   | `python -m uvicorn src.servicio.app:app --reload` |
| `.\run.ps1 evaluar` | `python -m src.evaluar`                           |
| `.\run.ps1 lint`    | `python -m ruff check src tests`                  |

---

## 🐳 Uso con Docker (Opcional)

> [!NOTE]
> **Docker no es necesario para desarrollar ni probar localmente en Windows.** Esta sección es únicamente si deseas empaquetar la aplicación en un contenedor para producción.

Si tienes **Docker Desktop** instalado y activo en Windows:

1. **Construir la imagen:**

   ```powershell
   docker build -t churn-api:latest .
   ```
2. **Ejecutar el contenedor:**

   ```powershell
   docker run --rm -p 8000:8000 churn-api:latest
   ```

---

## 🧪 CI/CD con GitHub Actions

El archivo `.github/workflows/ci.yml` ejecuta automáticamente el pipeline de integración continua al realizar un `git push` a tu repositorio de GitHub:

1. **Linting**: Valida calidad de código con `ruff`.
2. **Tests**: Ejecuta los tests unitarios con `pytest`.
3. **Smoke Train**: Entrena un modelo rápido con `data/muestra.csv` y sube las métricas como *artifact*.
4. **Build de Docker**: Construye y prueba la imagen Docker en los servidores de GitHub.

> **¡No necesitas Docker en tu máquina personal!** GitHub Actions construirá y probará la imagen Docker en la nube de forma totalmente transparente.

---

## 🛠️ Solución de Problemas Frecuentes (Windows Troubleshooting)

### ❌ Error: "La ejecución de scripts está deshabilitada en este sistema" (`PSSecurityException`)

**Causa:** Política predeterminada de seguridad en PowerShell.
**Solución:** Ejecuta antes de activar tu entorno:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### ❌ Error: `InconsistentVersionWarning` o fallo al deserializar `modelo.joblib`

**Causa:** Intentaste cargar un modelo `.joblib` creado en otra máquina con una versión distinta de Python o `scikit-learn`.
**Solución:** Entrena el modelo en tu propia máquina para generar un binario nativo:

```powershell
python -m src.entrenar
```

### ❌ Error: `[Errno 10048] address already in use` al hacer `serve`

**Causa:** El puerto 8000 ya está siendo utilizado por otra aplicación o un proceso anterior de Uvicorn.
**Solución:** Cambia el puerto al levantar la API:

```powershell
python -m uvicorn src.servicio.app:app --port 8050
```

### ❌ Error: "No module named src" al ejecutar comandos

**Causa:** Estás ejecutando el comando desde una subcarpeta en lugar de la raíz `plantilla-proyecto-mlops`.
**Solución:** Asegúrate de estar en `plantilla-proyecto-mlops` y usa la sintaxis `python -m src.entrenar`.

---

*Material del taller MLOps UNI 2026 — Optimizado para Windows y producción local.*
