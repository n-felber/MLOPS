ATHLETE ?= Mutaz Essa BARSHIM

.DEFAULT_GOAL := help

.PHONY: help build fetch features train inference ui mlflow upload-cloud-artifacts download-cloud-artifacts test clean

help:
	@echo "Available commands:"
	@echo "  make build                         Build Docker image"
	@echo "  make fetch                         Fetch raw World Athletics data"
	@echo "  make features                      Calculate features from raw data"
	@echo "  make train                         Run training pipeline"
	@echo "  make inference                     Run CLI inference with default athlete"
	@echo "  make inference ATHLETE=\"Name\"      Run CLI inference for selected athlete"
	@echo "  make ui                            Run Streamlit UI"
	@echo "  make mlflow                       Run local MLflow UI"
	@echo "  make upload-cloud-artifacts        Upload model/features to GCS"
	@echo "  make download-cloud-artifacts      Download model/features from GCS"
	@echo "  make test                          Run unit tests"
	@echo "  make clean                         Stop containers and remove orphan services"

build:
	docker compose build

fetch:
	docker compose run --rm fetch

features:
	docker compose run --rm features

train:
	docker compose run --rm train

inference:
	ATHLETE="$(ATHLETE)" docker compose run --rm inference

ui:
	docker compose up ui

mlflow:
	docker compose up mlflow

upload-cloud-artifacts:
	docker compose run --rm train upload-cloud-artifacts

download-cloud-artifacts:
	docker compose run --rm train download-cloud-artifacts

test:
	uv run pytest

clean:
	docker compose down --remove-orphans