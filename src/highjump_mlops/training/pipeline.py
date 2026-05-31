from numbers import Number
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from highjump_mlops.config import FEATURES_PATH, MODEL_PATH


FEATURE_COLUMNS = [
    "competition_year",
    "result_rank",
    "competition_mark",
    "results_score",
    "athlete_competition_number",
    "previous_competition_mark",
    "previous_result_rank",
    "previous_results_score",
    "days_since_previous_competition",
    "recent_3_competition_mark_mean",
    "recent_3_competition_mark_median",
    "recent_5_competition_mark_mean",
    "recent_5_competition_mark_median",
    "performance_change_from_previous",
    "season_result_count_so_far",
    "season_best_so_far",
]

TARGET_COLUMN = "target_next_competition_mark"

MLFLOW_TRACKING_URI = "file:mlruns"
MLFLOW_EXPERIMENT_NAME = "highjump-mlops"


def load_training_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_next_competition_date"] = pd.to_datetime(
        df["target_next_competition_date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "date",
            "target_next_competition_date",
        ]
    )

    return df


def train_model(df: pd.DataFrame) -> tuple[LinearRegression, dict[str, Any]]:
    train_df = df[df["date"] < pd.Timestamp("2024-01-01")]
    test_df = df[df["date"] >= pd.Timestamp("2024-01-01")]

    if train_df.empty:
        raise ValueError("Training data is empty.")

    if test_df.empty:
        raise ValueError("Test data is empty.")

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target_column": TARGET_COLUMN,
        "prediction_type": "next_competition_mark",
    }

    print(f"Training rows: {metrics['training_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"MAE: {metrics['mae']:.3f} m")
    print(f"RMSE: {metrics['rmse']:.3f} m")
    print(f"Target: {TARGET_COLUMN}")

    return model, metrics


def save_model(model: LinearRegression, metrics: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "target_column": TARGET_COLUMN,
        "prediction_type": "next_competition_mark",
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")


def log_to_mlflow(model: LinearRegression, metrics: dict[str, Any]) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, Number)
    }

    with mlflow.start_run(run_name="linear-regression-next-competition") as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "model_type": "LinearRegression",
                "target_column": TARGET_COLUMN,
                "prediction_type": "next_competition_mark",
                "feature_count": len(FEATURE_COLUMNS),
                "train_split": "date < 2024-01-01",
                "test_split": "date >= 2024-01-01",
                "feature_store_path": str(FEATURES_PATH),
                "deployed_model_path": str(MODEL_PATH),
            }
        )

        for index, feature in enumerate(FEATURE_COLUMNS, start=1):
            mlflow.log_param(f"feature_{index}", feature)

        mlflow.log_metrics(numeric_metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model_package",
        )

        latest_run_path = MODEL_PATH.parent / "latest_mlflow_run.txt"
        latest_run_path.write_text(f"{run_id}\n", encoding="utf-8")

        mlflow.log_artifact(
            str(latest_run_path),
            artifact_path="deployment",
        )

        print(f"Logged MLflow run: {run_id}")
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

        return run_id


def main() -> None:
    df = load_training_data()
    model, metrics = train_model(df)
    save_model(model, metrics)
    log_to_mlflow(model, metrics)


if __name__ == "__main__":
    main()