import pandas as pd


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
