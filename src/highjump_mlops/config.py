from pathlib import Path


YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

RAW_DIR = Path("data/raw")
FEATURES_PATH = Path("data/features/highjump_features.parquet")
MODEL_PATH = Path("models/highjump_model.joblib")


def toplist_url(year: int) -> str:
    return (
        "https://worldathletics.org/records/toplists/jumps/high-jump/"
        f"outdoor/men/senior/{year}"
    )
