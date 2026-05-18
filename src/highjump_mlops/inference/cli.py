import argparse
from typing import Any

from highjump_mlops.inference.service import get_athlete_history, list_predictable_athletes, predict_for_athlete


def print_prediction(result: dict[str, Any]) -> None:
    print("\nPrediction result:")
    print(f"Athlete: {result['athlete']}")
    print(f"Latest year: {result['latest_year']}")
    print(f"Latest season best: {result['latest_season_best']:.2f} m")
    print(
        "Predicted next season best: "
        f"{result['prediction_next_season_best']:.2f} m"
    )
    print(f"Latest season rank: {result['latest_season_rank']}")
    print(f"Previous season best: {result['previous_season_best']}")
    print(f"Performance change: {result['performance_change']}")
    print(f"Days since season best: {result['days_since_season_best']}")


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
    print(get_athlete_history(athlete).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
