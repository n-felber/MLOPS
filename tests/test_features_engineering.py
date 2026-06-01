import pandas as pd
import pytest

from highjump_mlops.features.engineering import build_features


def make_raw_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": [3, 1, 2, 5],
            "mark": [2.00, 2.10, 2.05, 1.90],
            "competitor": [
                "Test ATHLETE",
                "Test ATHLETE",
                "Test ATHLETE",
                "Other ATHLETE",
            ],
            "dob": [
                "01 JAN 2000",
                "01 JAN 2000",
                "01 JAN 2000",
                "01 JAN 2001",
            ],
            "pos": ["3", "1", "2", "5"],
            "venue": [
                "Venue A",
                "Venue B",
                "Venue C",
                "Venue D",
            ],
            "date": [
                "2024-01-01",
                "2024-01-10",
                "2024-01-20",
                "2024-01-05",
            ],
            "results_score": [1000, 1050, 1020, 900],
            "year": [2024, 2024, 2024, 2024],
            "source_page": [1, 1, 1, 1],
        }
    )


def test_build_features_creates_lag_rolling_and_target_features() -> None:
    features = build_features(make_raw_results())

    athlete_rows = (
        features[features["athlete"] == "Test ATHLETE"]
        .sort_values("date")
        .reset_index(drop=True)
    )

    assert len(athlete_rows) == 3

    first_row = athlete_rows.iloc[0]
    second_row = athlete_rows.iloc[1]
    third_row = athlete_rows.iloc[2]

    assert first_row["competition_mark"] == pytest.approx(2.00)
    assert pd.isna(first_row["previous_competition_mark"])
    assert first_row["target_next_competition_mark"] == pytest.approx(2.10)

    assert second_row["competition_mark"] == pytest.approx(2.10)
    assert second_row["previous_competition_mark"] == pytest.approx(2.00)
    assert second_row["days_since_previous_competition"] == 9
    assert second_row["recent_3_competition_mark_mean"] == pytest.approx(2.00)
    assert second_row["performance_change_from_previous"] == pytest.approx(0.10)
    assert second_row["target_next_competition_mark"] == pytest.approx(2.05)

    assert third_row["competition_mark"] == pytest.approx(2.05)
    assert third_row["previous_competition_mark"] == pytest.approx(2.10)
    assert third_row["days_since_previous_competition"] == 10
    assert third_row["recent_3_competition_mark_mean"] == pytest.approx(2.05)
    assert third_row["season_best_so_far"] == pytest.approx(2.10)
    assert pd.isna(third_row["target_next_competition_mark"])


def test_build_features_keeps_athletes_separate() -> None:
    features = build_features(make_raw_results())

    other_athlete_row = features[features["athlete"] == "Other ATHLETE"].iloc[0]

    assert other_athlete_row["competition_mark"] == pytest.approx(1.90)
    assert pd.isna(other_athlete_row["previous_competition_mark"])
    assert pd.isna(other_athlete_row["target_next_competition_mark"])
    assert other_athlete_row["athlete_competition_number"] == 1
