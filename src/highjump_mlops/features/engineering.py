import pandas as pd


def build_features(results: pd.DataFrame) -> pd.DataFrame:
    competition_results = results.rename(
        columns={
            "competitor": "athlete",
            "rank": "result_rank",
            "mark": "competition_mark",
        }
    ).copy()

    competition_results["date"] = pd.to_datetime(
        competition_results["date"],
        errors="coerce",
    )

    competition_results["competition_year"] = competition_results["date"].dt.year

    competition_results = competition_results.dropna(
        subset=[
            "athlete",
            "competition_mark",
            "date",
            "results_score",
        ]
    )

    competition_results["athlete"] = competition_results["athlete"].astype(str).str.strip()

    competition_results = competition_results.drop_duplicates(
        subset=[
            "athlete",
            "date",
            "venue",
            "competition_mark",
            "pos",
        ]
    )

    competition_results = competition_results.sort_values(
        [
            "athlete",
            "date",
            "competition_mark",
            "results_score",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    athlete_group = competition_results.groupby("athlete", group_keys=False)

    competition_results["athlete_competition_number"] = (
        athlete_group.cumcount() + 1
    )

    competition_results["previous_competition_mark"] = athlete_group[
        "competition_mark"
    ].shift(1)

    competition_results["previous_result_rank"] = athlete_group[
        "result_rank"
    ].shift(1)

    competition_results["previous_results_score"] = athlete_group[
        "results_score"
    ].shift(1)

    previous_competition_date = athlete_group["date"].shift(1)

    competition_results["days_since_previous_competition"] = (
        competition_results["date"] - previous_competition_date
    ).dt.days

    competition_results["recent_3_competition_mark_mean"] = athlete_group[
        "competition_mark"
    ].transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())

    competition_results["recent_3_competition_mark_median"] = athlete_group[
        "competition_mark"
    ].transform(lambda values: values.shift(1).rolling(3, min_periods=1).median())

    competition_results["recent_5_competition_mark_mean"] = athlete_group[
        "competition_mark"
    ].transform(lambda values: values.shift(1).rolling(5, min_periods=1).mean())

    competition_results["recent_5_competition_mark_median"] = athlete_group[
        "competition_mark"
    ].transform(lambda values: values.shift(1).rolling(5, min_periods=1).median())

    competition_results["performance_change_from_previous"] = (
        competition_results["competition_mark"]
        - competition_results["previous_competition_mark"]
    )

    season_group = competition_results.groupby(
        [
            "athlete",
            "competition_year",
        ],
        group_keys=False,
    )

    competition_results["season_result_count_so_far"] = (
        season_group.cumcount() + 1
    )

    competition_results["season_best_so_far"] = season_group[
        "competition_mark"
    ].cummax()

    competition_results["target_next_competition_mark"] = athlete_group[
        "competition_mark"
    ].shift(-1)

    competition_results["target_next_competition_date"] = athlete_group[
        "date"
    ].shift(-1)

    competition_results["days_until_next_competition"] = (
        competition_results["target_next_competition_date"]
        - competition_results["date"]
    ).dt.days

    return competition_results
