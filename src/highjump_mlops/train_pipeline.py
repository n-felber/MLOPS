from pathlib import Path

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
    "previous_season_best",
    "previous_results_score",
    "performance_change",
]

TARGET_COLUMN = "target_next_season_best"


def load_training_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    return df


def train_model(df: pd.DataFrame) -> LinearRegression:
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

    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"MAE: {mae:.3f} m")
    print(f"RMSE: {rmse:.3f} m")

    return model


def save_model(model: LinearRegression) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")


def main() -> None:
    df = load_training_data()
    model = train_model(df)
    save_model(model)


if __name__ == "__main__":
    main()