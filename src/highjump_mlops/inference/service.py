from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from highjump_mlops.config import FEATURES_PATH, MODEL_PATH



def load_features(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}. "
            "Run the feature pipeline inside Docker first:\n\n"
            "make features"
        )

    df = pd.read_parquet(features_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def load_model_package(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run the training pipeline inside Docker first:\n\n"
            "make train"
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

    missing_columns = [column for column in feature_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing feature columns in data: {missing_columns}")

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
        "date",
        "venue",
        "competition_mark",
        "result_rank",
        "previous_competition_mark",
        "recent_3_competition_mark_mean",
        "recent_5_competition_mark_mean",
        "performance_change_from_previous",
        "days_since_previous_competition",
        "season_best_so_far",
        "target_next_competition_mark",
    ]

    existing_columns = [col for col in history_columns if col in athlete_rows.columns]

    return (
        athlete_rows[existing_columns]
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )


def optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None

    return float(value)


def optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None

    return int(value)


def optional_str(value: Any) -> str | None:
    if pd.isna(value):
        return None

    return str(value)


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
            "The athlete may not have enough previous competitions for prediction."
        )

    latest_row = usable_rows.sort_values("date").iloc[-1]

    x = latest_row[feature_columns].to_frame().T
    prediction = float(model.predict(x)[0])

    latest_date = pd.to_datetime(latest_row["date"], errors="coerce")
    
    today = pd.Timestamp.today().normalize()
    days_since_latest_competition = (
        None if pd.isna(latest_date) else int((today - latest_date).days)
    )

    return {
        "athlete": athlete,
        "prediction_next_competition_mark": prediction,
        "model_name": package.get("model_name"),
        "model_type": package.get("model_type"),
        "latest_date": None if pd.isna(latest_date) else latest_date.date().isoformat(),
        "latest_venue": optional_str(latest_row.get("venue")),
        "latest_competition_mark": float(latest_row["competition_mark"]),
        "latest_result_rank": optional_int(latest_row.get("result_rank")),
        "previous_competition_mark": optional_float(
            latest_row.get("previous_competition_mark")
        ),
        "recent_3_competition_mark_mean": optional_float(
            latest_row.get("recent_3_competition_mark_mean")
        ),
        "recent_5_competition_mark_mean": optional_float(
            latest_row.get("recent_5_competition_mark_mean")
        ),
        "performance_change_from_previous": optional_float(
            latest_row.get("performance_change_from_previous")
        ),
        "days_since_previous_competition": optional_int(
            latest_row.get("days_since_previous_competition")
        ),
        "season_best_so_far": optional_float(
            latest_row.get("season_best_so_far")
        ),
        "season_result_count_so_far": optional_int(
            latest_row.get("season_result_count_so_far")
        ),
        "metrics": package.get("metrics", {}),
        "days_since_latest_competition": days_since_latest_competition,
    }
