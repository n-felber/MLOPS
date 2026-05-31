# High Jump Live ML System (MLOPS - Project)

This project is a live machine learning system for predicting men's outdoor high jump performance.

The current baseline predicts an athlete's **next competition mark** using dynamic World Athletics result data.

## How to run with Docker

Build the Docker image:

```bash
make build
```

Run the feature pipeline:

```bash
make features
```

Run the training pipeline:

```bash
make train
```

Run CLI inference with the default athlete:

```bash
make inference
```

Run CLI inference for a selected athlete:

```bash
make inference ATHLETE="Gianmarco TAMBERI"
```

Run the Streamlit UI:

```bash
make ui
```

Then open:

```text
http://localhost:8501
```

Run the MLflow UI:

```bash
make mlflow
```

Then open:

```text
http://localhost:5001
```

## Architecture

The project follows the FTI architecture:

```text
Feature Pipeline -> Parquet Feature Store -> Training Pipeline -> Model Artifact / MLflow -> Inference/UI
```

### Feature Pipeline

The feature pipeline scrapes World Athletics toplist pages, parses the results, builds athlete-level competition features, and saves them as a Parquet feature store.

Output:

```text
data/features/highjump_features.parquet
```

### Training Pipeline

The training pipeline reads the feature store, trains a regression model, evaluates it with MAE and RMSE, logs the run to MLflow, and saves the trained model package.

Output:

```text
models/highjump_model.joblib
```

MLflow outputs:

```text
mlruns/
models/latest_mlflow_run.txt
```

### Inference/UI

The inference pipeline loads the saved model and latest features, predicts the selected athlete's next competition mark, and serves the result through a Streamlit UI.


## Current scope

The original proposal targeted next-competition prediction. During implementation, detailed per-attempt data was not yet reliably available from the source, so the current baseline predicts the next available competition mark from dynamic World Athletics result data. The goal remains to extend the system toward richer competition-level features once the required data can be accessed and processed reliably.

## Model evaluation

The Streamlit UI displays:

```text
Predicted next competition mark
Latest competition mark
Latest result rank
Recent athlete history
MAE
RMSE
Training rows
Test rows
```

## Tech stack

* Python
* Docker
* uv
* pandas
* scikit-learn
* Streamlit
* Parquet
* joblib
* MLflow
