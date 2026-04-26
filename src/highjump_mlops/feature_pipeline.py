from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Current limitation:
# each yearly toplist currently loads only the first page / top 100 results.
# Later I will add pagination to collect deeper rankings.
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

RAW_DIR = Path("data/raw")
FEATURES_PATH = Path("data/features/highjump_features.parquet")


def toplist_url(year: int) -> str:
    return (
        "https://worldathletics.org/records/toplists/jumps/high-jump/"
        f"outdoor/men/senior/{year}"
    )


def fetch_html(year: int) -> str:
    response = requests.get(
        toplist_url(year),
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def parse_toplist(html: str, year: int) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))

    if not tables:
        raise RuntimeError(f"No tables found for {year}.")

    df = tables[0]
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]

    df = df[["rank", "mark", "competitor", "date", "results_score"]].copy()
    df["year"] = year

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["mark"] = pd.to_numeric(df["mark"], errors="coerce")
    df["results_score"] = pd.to_numeric(df["results_score"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")

    df = df.dropna(subset=["rank", "mark", "competitor", "date", "results_score"])

    return df


def build_features(results: pd.DataFrame) -> pd.DataFrame:
    athlete_seasons = (
        results.sort_values(["competitor", "year", "mark"], ascending=[True, True, False])
        .groupby(["competitor", "year"], as_index=False)
        .first()
    )

    athlete_seasons = athlete_seasons.rename(
        columns={
            "competitor": "athlete",
            "mark": "season_best",
            "rank": "season_rank",
        }
    )

    athlete_seasons = athlete_seasons.sort_values(["athlete", "year"])

    athlete_seasons["previous_season_best"] = athlete_seasons.groupby("athlete")[
        "season_best"
    ].shift(1)

    athlete_seasons["previous_results_score"] = athlete_seasons.groupby("athlete")[
        "results_score"
    ].shift(1)

    athlete_seasons["performance_change"] = (
        athlete_seasons["season_best"] - athlete_seasons["previous_season_best"]
    )

    athlete_seasons["days_since_season_best"] = (
        pd.Timestamp.today().normalize() - athlete_seasons["date"]
    ).dt.days

    athlete_seasons["target_next_season_best"] = athlete_seasons.groupby("athlete")[
        "season_best"
    ].shift(-1)

    return athlete_seasons


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    yearly_results = []

    for year in YEARS:
        html = fetch_html(year)
        (RAW_DIR / f"world_athletics_toplist_{year}.html").write_text(html)

        year_results = parse_toplist(html, year)
        yearly_results.append(year_results)

        print(f"Fetched {len(year_results)} rows for {year}")

    results = pd.concat(yearly_results, ignore_index=True)
    features = build_features(results)

    features.to_parquet(FEATURES_PATH, index=False)

    print(f"Saved {len(features)} feature rows to {FEATURES_PATH}")
    print(features.head())


if __name__ == "__main__":
    main()