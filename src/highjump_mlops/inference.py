from pathlib import Path
from typing import Any
import argparse

import joblib
import pandas as pd

from highjump_mlops.config import FEATURES_PATH, MODEL_PATH



def load_features(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}. "
            "Run the feature pipeline inside Docker first:\n\n"
            'docker run --rm -v "$PWD/data:/app/data" highjump-mlops feature-pipeline'
        )

    return pd.read_parquet(features_path)


def load_model_package(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run the training pipeline inside Docker first:\n\n"
            'docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" '
            "highjump-mlops train-pipeline"
        )

    package = joblib.load(model_path)

    if not isinstance(package, dict):
        raise TypeError("Expected the saved model file to contain a dictionary.")

    if "model" not in package:
        raise KeyError("Model package is missing the key: model")

    if "feature_columns" not in package:
        raise KeyError("Model package is missing the key: feature_columns")

    return package


def list_available_athletes(features_path: Path = FEATURES_PATH) -> list[str]:
    df = load_features(features_path)

    return (
        df["athlete"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )


def list_predictable_athletes(features_path: Path = FEATURES_PATH, model_path: Path = MODEL_PATH) -> list[str]:
    df = load_features(features_path)
    package = load_model_package(model_path)
    feature_columns = package["feature_columns"]

    usable_df = df.dropna(subset=feature_columns)

    return (
        usable_df["athlete"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )


def get_athlete_history(athlete: str, features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = load_features(features_path)

    athlete_rows = df[df["athlete"] == athlete].copy()

    if athlete_rows.empty:
        raise ValueError(f"No rows found for athlete: {athlete}")

    history_columns = [
        "year",
        "season_rank",
        "season_best",
        "previous_season_best",
        "recent_3_season_best_mean",
        "performance_change",
        "days_since_season_best",
    ]

    existing_columns = [col for col in history_columns if col in athlete_rows.columns]

    return (
        athlete_rows[existing_columns]
        .sort_values("year", ascending=False)
        .reset_index(drop=True)
    )


def predict_for_athlete(athlete: str, features_path: Path = FEATURES_PATH, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    df = load_features(features_path)
    package = load_model_package(model_path)

    model = package["model"]
    feature_columns = package["feature_columns"]

    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns in data: {missing_columns}")

    athlete_rows = df[df["athlete"] == athlete].copy()

    if athlete_rows.empty:
        raise ValueError(f"No rows found for athlete: {athlete}")

    usable_rows = athlete_rows.dropna(subset=feature_columns).copy()

    if usable_rows.empty:
        raise ValueError(
            f"No usable feature rows found for athlete: {athlete}. "
            "The athlete may not have enough previous seasons for prediction."
        )

    latest_row = usable_rows.sort_values("year").iloc[-1]

    x = latest_row[feature_columns].to_frame().T
    prediction = float(model.predict(x)[0])

    return {
        "athlete": athlete,
        "prediction_next_season_best": prediction,
        "latest_year": int(latest_row["year"]),
        "latest_season_best": float(latest_row["season_best"]),
        "latest_season_rank": int(latest_row["season_rank"]),
        "previous_season_best": (
            None
            if pd.isna(latest_row["previous_season_best"])
            else float(latest_row["previous_season_best"])
        ),
        "performance_change": (
            None
            if pd.isna(latest_row["performance_change"])
            else float(latest_row["performance_change"])
        ),
        "days_since_season_best": int(latest_row["days_since_season_best"]),
    }


def print_prediction(result: dict[str, Any]) -> None:
    print("\nPrediction result:")
    print(f"Athlete: {result['athlete']}")
    print(f"Latest year: {result['latest_year']}")
    print(f"Latest season best: {result['latest_season_best']:.2f} m")
    print(f"Predicted next season best: {result['prediction_next_season_best']:.2f} m")
    print(f"Latest season rank: {result['latest_season_rank']}")
    print(f"Previous season best: {result['previous_season_best']}")
    print(f"Performance change: {result['performance_change']}")
    print(f"Days since season best: {result['days_since_season_best']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for a men's outdoor high jump athlete.")
    parser.add_argument("athlete", nargs="?", help="Exact athlete name. If omitted, the first predictable athlete is used.")

    args = parser.parse_args()

    predictable_athletes = list_predictable_athletes()

    print(f"Predictable athletes: {len(predictable_athletes)}")
    print("First 10 predictable athletes:")
    for athlete_name in predictable_athletes[:10]:
        print("-", athlete_name)

    athlete = args.athlete or predictable_athletes[0]

    print(f"\nSelected athlete: {athlete}")

    result = predict_for_athlete(athlete)
    print_prediction(result)

    print("\nRecent athlete history:")
    print(get_athlete_history(athlete).head(5).to_string(index=False))


if __name__ == "__main__":
    main()