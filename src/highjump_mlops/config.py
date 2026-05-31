from datetime import date
from pathlib import Path


YEARS_BACK = 6
CURRENT_YEAR = date.today().year
START_YEAR = CURRENT_YEAR - YEARS_BACK
YEARS = list(range(START_YEAR, CURRENT_YEAR + 1))

RAW_DIR = Path("data/raw")
FEATURES_PATH = Path("data/features/highjump_features.parquet")
MODEL_PATH = Path("models/highjump_model.joblib")


def toplist_url(year: int, page: int) -> str:
    return (
        "https://worldathletics.org/records/toplists/jumps/high-jump/"
        f"outdoor/men/senior/{year}"
        f"?ageCategory=senior"
        f"&bestResultsOnly=false"
        f"&eventId=10229615"
        f"&maxResultsByCountry=all"
        f"&page={page}"
        f"&regionType=world"
    )