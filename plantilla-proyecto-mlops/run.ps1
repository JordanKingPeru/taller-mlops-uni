<#
.SYNOPSIS
    Script helper de PowerShell para emular Makefile en Windows.
.DESCRIPTION
    Permite ejecutar comandos habituales de MLOps en Windows de forma fácil:
    .\run.ps1 train
    .\run.ps1 test
    .\run.ps1 serve [puerto]
    .\run.ps1 lint
    .\run.ps1 evaluar
#>

param (
    [Parameter(Position=0)]
    [string]$Command = "help",

    [Parameter(Position=1)]
    [int]$Port = 8000
)

# Forzar codificación UTF-8 en PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

switch ($Command.ToLower()) {
    "train" {
        Write-Host "-> Entrenando el modelo y evaluando Quality Gate..." -ForegroundColor Cyan
        python -m src.entrenar
    }
    "test" {
        Write-Host "-> Ejecutando pruebas unitarias con pytest..." -ForegroundColor Cyan
        python -m pytest tests/ -v
    }
    "serve" {
        Write-Host "-> Levantando API de predicción (FastAPI + Uvicorn) en el puerto $Port..." -ForegroundColor Cyan
        Write-Host "👉 Documentación interactiva en: http://127.0.0.1:$Port/docs" -ForegroundColor Green
        python -m uvicorn src.servicio.app:app --reload --host 127.0.0.1 --port $Port
    }
    "lint" {
        Write-Host "-> Verificando calidad de código con ruff..." -ForegroundColor Cyan
        python -m ruff check src tests
    }
    "evaluar" {
        Write-Host "-> Evaluando modelo guardado..." -ForegroundColor Cyan
        python -m src.evaluar
    }
    "docker-build" {
        Write-Host "-> Construyendo imagen Docker (Opcional)..." -ForegroundColor Cyan
        docker build -t churn-api:latest .
    }
    "docker-run" {
        Write-Host "-> Ejecutando contenedor Docker (Opcional)..." -ForegroundColor Cyan
        docker run --rm -p ${Port}:8000 churn-api:latest
    }
    default {
        Write-Host "Helper de comandos MLOps para Windows (PowerShell)" -ForegroundColor Yellow
        Write-Host "Uso: .\run.ps1 [comando] [puerto_opcional]" -ForegroundColor White
        Write-Host ""
        Write-Host "Comandos disponibles:" -ForegroundColor Gray
        Write-Host "  train        - Entrena el modelo localmente y valida el Quality Gate" -ForegroundColor White
        Write-Host "  test         - Ejecuta los tests del contrato de datos, modelo y API" -ForegroundColor White
        Write-Host "  serve [port] - Levanta la API local de FastAPI (ej: .\run.ps1 serve 8050)" -ForegroundColor White
        Write-Host "  evaluar      - Evalúa un modelo previamente entrenado" -ForegroundColor White
        Write-Host "  lint         - Revisa el código con ruff" -ForegroundColor White
        Write-Host "  docker-build - (Opcional) Construye la imagen Docker" -ForegroundColor White
        Write-Host "  docker-run   - (Opcional) Corre el contenedor en el puerto $Port" -ForegroundColor White
    }
}
