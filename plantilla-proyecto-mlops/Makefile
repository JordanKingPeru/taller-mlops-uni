.PHONY: train test serve lint docker-build docker-run evaluar

train:
	python -m src.entrenar

evaluar:
	python -m src.evaluar

test:
	python -m pytest tests/ -v

lint:
	ruff check src tests

serve:
	uvicorn src.servicio.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t churn-api:latest .

docker-run:
	docker run --rm -p 8000:8000 churn-api:latest
