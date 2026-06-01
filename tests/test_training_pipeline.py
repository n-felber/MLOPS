import pandas as pd
import pytest

from highjump_mlops.training.pipeline import split_train_test_by_time


def make_training_rows() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")

    return pd.DataFrame(
        {
            "date": dates,
            "target_next_competition_date": dates + pd.Timedelta(days=1),
        }
    )


def test_split_train_test_by_time_uses_future_rows_for_test_set() -> None:
    df = make_training_rows()

    train_df, test_df, cutoff_date = split_train_test_by_time(df, test_size=0.3)

    assert cutoff_date == pd.Timestamp("2024-01-08")

    assert not train_df.empty
    assert not test_df.empty

    assert train_df["date"].max() < cutoff_date
    assert train_df["target_next_competition_date"].max() < cutoff_date
    assert test_df["date"].min() >= cutoff_date


def test_split_train_test_by_time_rejects_empty_dataframe() -> None:
    empty_df = pd.DataFrame(
        {
            "date": [],
            "target_next_competition_date": [],
        }
    )

    with pytest.raises(ValueError, match="empty training dataframe"):
        split_train_test_by_time(empty_df)


def test_split_train_test_by_time_rejects_invalid_test_size() -> None:
    df = make_training_rows()

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        split_train_test_by_time(df, test_size=1.0)
