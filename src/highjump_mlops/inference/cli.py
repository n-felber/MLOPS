import argparse
from typing import Any

from highjump_mlops.inference.service import get_athlete_history, list_predictable_athletes, predict_for_athlete


def format_optional_height(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value:.2f} m"


def print_prediction(result: dict[str, Any]) -> None:
    print("\nPrediction result:")
    print(f"Athlete: {result['athlete']}")
    print(f"Latest competition date: {result['latest_date']}")
    print(f"Latest venue: {result['latest_venue']}")
    print(f"Latest competition mark: {result['latest_competition_mark']:.2f} m")
    print(
        "Predicted next competition mark: "
        f"{result['prediction_next_competition_mark']:.2f} m"
    )
    print(f"Latest result rank: {result['latest_result_rank']}")
    print(
        "Previous competition mark: "
        f"{format_optional_height(result['previous_competition_mark'])}"
    )
    print(
        "Recent 3-competition mean: "
        f"{format_optional_height(result['recent_3_competition_mark_mean'])}"
    )
    print(
        "Recent 5-competition mean: "
        f"{format_optional_height(result['recent_5_competition_mark_mean'])}"
    )
    print(
        "Performance change from previous: "
        f"{format_optional_height(result['performance_change_from_previous'])}"
    )
    print(
        "Days since previous competition: "
        f"{result['days_since_previous_competition']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference for a men's outdoor high jump athlete."
    )
    parser.add_argument(
        "athlete",
        nargs="?",
        help="Exact athlete name. If omitted, the first predictable athlete is used.",
    )

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
    print(get_athlete_history(athlete).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
