from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


FEATURES_PATH = Path("data/features/highjump_features.parquet")
MODEL_PATH = Path("models/highjump_model.joblib")

FEATURE_COLUMNS = [
    "season_rank",
    "season_best",
    "results_score",
    "season_result_count",
    "previous_season_best",
    "previous_season_rank",
    "previous_results_score",
    "recent_3_season_best_mean",
    "recent_3_season_best_median",
    "performance_change",
]

TARGET_COLUMN = "target_next_season_best"


def load_training_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    return df


def train_model(df: pd.DataFrame) -> tuple[LinearRegression, dict[str, Any]]:
    train_df = df[df["year"] <= 2023]
    test_df = df[df["year"] >= 2024]

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
    }

    print(f"Training rows: {metrics['training_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"MAE: {metrics['mae']:.3f} m")
    print(f"RMSE: {metrics['rmse']:.3f} m")

    return model, metrics


def save_model(model: LinearRegression, metrics: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "target_column": TARGET_COLUMN,
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")


def main() -> None:
    df = load_training_data()
    model, metrics = train_model(df)
    save_model(model, metrics)


if __name__ == "__main__":
    main()