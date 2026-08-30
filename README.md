# 🎓 Taller MLOps en la Práctica: Del Notebook a Producción

![UNI](https://img.shields.io/badge/UNI-Especializaci%C3%B3n%20Ciencia%20de%20Datos-red)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Windows Friendly](https://img.shields.io/badge/Windows-100%25%20Compatible-brightgreen)

Repositorio oficial del taller **MLOps en la práctica: del notebook a producción**, dictado en la Universidad Nacional de Ingeniería (UNI).

---

## 📂 Contenido del Repositorio

| Carpeta / Archivo | Descripción |
| :--- | :--- |
| **[`plantilla-proyecto-mlops/`](./plantilla-proyecto-mlops/)** | **Proyecto principal del taller**: Estructura modular para producción (`src/`, `tests/`, `params.yaml`, `FastAPI`, `GitHub Actions`). |
| **[`notebooks/`](./notebooks/)** | Notebooks interactivos del curso (reproducibilidad, MLflow tracking, registry/serving y monitoreo de drift). |
| **[`guias/`](./guias/)** | Guías complementarias en formato PDF y DOCX para el alumno y madurez MLOps. |

---

## 🚀 Inicio Rápido en Windows (Sin Docker)

Para empezar a trabajar inmediatamente en tu máquina con **Windows (PowerShell)**:

1. **Ingresa a la plantilla del proyecto**:
   ```powershell
   cd plantilla-proyecto-mlops
   ```

2. **Revisa la guía detallada de instalación y uso**:
   Abre el archivo [plantilla-proyecto-mlops/README.md](./plantilla-proyecto-mlops/README.md) para seguir el paso a paso de configuración en Windows sin necesidad de Docker ni de versiones específicas de Python.

3. **Ejecución rápida en 3 pasos**:
   ```powershell
   # 1. Crear entorno e instalar dependencias
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   # 2. Entrenar el modelo (recalcula el modelo.joblib para tu versión de Python)
   python -m src.entrenar

   # 3. Levantar la API localmente
   python -m uvicorn src.servicio.app:app --reload
   ```
   Abre tu navegador en **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** para interactuar con la API.

---

*Material educativo del Taller MLOps — UNI 2026.*
