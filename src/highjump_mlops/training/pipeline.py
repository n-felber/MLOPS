from numbers import Number
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from highjump_mlops.config import FEATURES_PATH, MODEL_PATH


FEATURE_COLUMNS: list[str] = [
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

TARGET_COLUMN: str = "target_next_competition_mark"

MLFLOW_TRACKING_URI: str = "file:mlruns"
MLFLOW_EXPERIMENT_NAME: str = "highjump-mlops"

TEST_SIZE: float = 0.2


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


def split_train_test_by_time(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if df.empty:
        raise ValueError("Cannot split an empty training dataframe.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    sorted_df = df.sort_values("date").reset_index(drop=True)

    split_index = int(len(sorted_df) * (1 - test_size))

    if split_index <= 0 or split_index >= len(sorted_df):
        raise ValueError(
            "Could not create a valid train/test split. "
            f"Rows available: {len(sorted_df)}, test_size: {test_size}"
        )

    cutoff_date = sorted_df.loc[split_index, "date"]

    train_df = sorted_df[
        (sorted_df["date"] < cutoff_date)
        & (sorted_df["target_next_competition_date"] < cutoff_date)
    ].copy()

    test_df = sorted_df[sorted_df["date"] >= cutoff_date].copy()

    if train_df.empty:
        raise ValueError("Training data is empty after time-based split.")

    if test_df.empty:
        raise ValueError("Test data is empty after time-based split.")

    return train_df, test_df, cutoff_date


def get_candidate_models() -> dict[str, Any]:
    return {
        "linear-regression": LinearRegression(),
        "random-forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "hist-gradient-boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }


def train_model(model_name: str, model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> tuple[Any, dict[str, Any]]:
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    metrics = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "mae": float(mae),
        "rmse": float(rmse),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cutoff_date": cutoff_date.date().isoformat(),
        "test_size": TEST_SIZE,
        "target_column": TARGET_COLUMN,
        "prediction_type": "next_competition_mark",
    }

    print(f"\nModel: {model_name}")
    print(f"Training rows: {metrics['training_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Cutoff date: {metrics['cutoff_date']}")
    print(f"MAE: {metrics['mae']:.4f} m")
    print(f"RMSE: {metrics['rmse']:.4f} m")
    print(f"Target: {TARGET_COLUMN}")

    return model, metrics


def save_model(model: Any, metrics: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "model_name": metrics["model_name"],
        "model_type": metrics["model_type"],
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "target_column": TARGET_COLUMN,
        "prediction_type": "next_competition_mark",
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")


def log_to_mlflow(model: Any, metrics: dict[str, Any], is_deployed: bool) -> str:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, Number)
    }

    with mlflow.start_run(run_name=f"{metrics['model_name']}-next-competition") as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "model_name": metrics["model_name"],
                "model_type": metrics["model_type"],
                "target_column": TARGET_COLUMN,
                "prediction_type": "next_competition_mark",
                "feature_count": len(FEATURE_COLUMNS),
                "train_split": (
                    f"date < {metrics['cutoff_date']} and "
                    f"target_next_competition_date < {metrics['cutoff_date']}"
                ),
                "test_split": f"date >= {metrics['cutoff_date']}",
                "test_size": metrics["test_size"],
                "feature_store_path": str(FEATURES_PATH),
                "deployed_model_path": str(MODEL_PATH),
                "deployment_status": "deployed" if is_deployed else "candidate",
            }
        )

        for index, feature in enumerate(FEATURE_COLUMNS, start=1):
            mlflow.log_param(f"feature_{index}", feature)

        mlflow.log_metrics(numeric_metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        if is_deployed:
            latest_run_path = MODEL_PATH.parent / "latest_mlflow_run.txt"
            latest_run_path.write_text(f"{run_id}\n", encoding="utf-8")

            mlflow.log_artifact(
                str(MODEL_PATH),
                artifact_path="model_package",
            )

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
    train_df, test_df, cutoff_date = split_train_test_by_time(df)

    trained_models: list[dict[str, Any]] = []

    for model_name, candidate_model in get_candidate_models().items():
        model, metrics = train_model(
            model_name=model_name,
            model=candidate_model,
            train_df=train_df,
            test_df=test_df,
            cutoff_date=cutoff_date,
        )

        trained_models.append(
            {
                "model": model,
                "metrics": metrics,
            }
        )

    best_result = min(
        trained_models,
        key=lambda result: result["metrics"]["mae"],
    )

    best_model = best_result["model"]
    best_metrics = best_result["metrics"]

    print("\nModel comparison:")
    for result in sorted(trained_models, key=lambda result: result["metrics"]["mae"]):
        metrics = result["metrics"]
        print(
            f"- {metrics['model_name']}: "
            f"MAE={metrics['mae']:.4f} m, "
            f"RMSE={metrics['rmse']:.4f} m"
        )

    print(
        "\nSelected best model: "
        f"{best_metrics['model_name']} "
        f"(MAE={best_metrics['mae']:.4f} m)"
    )

    save_model(best_model, best_metrics)

    for result in trained_models:
        metrics = result["metrics"]

        log_to_mlflow(
            model=result["model"],
            metrics=metrics,
            is_deployed=metrics["model_name"] == best_metrics["model_name"],
        )


if __name__ == "__main__":
    main()