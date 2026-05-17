import pandas as pd


def build_features(results: pd.DataFrame) -> pd.DataFrame:
    season_counts = (
        results.groupby(["competitor", "year"])
        .size()
        .reset_index(name="season_result_count")
    )

    athlete_seasons = (
        results.sort_values(
            ["competitor", "year", "mark", "results_score"],
            ascending=[True, True, False, False],
        )
        .groupby(["competitor", "year"], as_index=False)
        .first()
    )

    athlete_seasons = athlete_seasons.merge(
        season_counts,
        on=["competitor", "year"],
        how="left",
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

    athlete_seasons["previous_season_rank"] = athlete_seasons.groupby("athlete")[
        "season_rank"
    ].shift(1)

    athlete_seasons["previous_results_score"] = athlete_seasons.groupby("athlete")[
        "results_score"
    ].shift(1)

    athlete_seasons["recent_3_season_best_mean"] = athlete_seasons.groupby("athlete")[
        "season_best"
    ].transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())

    athlete_seasons["recent_3_season_best_median"] = athlete_seasons.groupby("athlete")[
        "season_best"
    ].transform(lambda values: values.shift(1).rolling(3, min_periods=1).median())

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
