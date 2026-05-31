import pandas as pd

from highjump_mlops.config import FEATURES_PATH, RAW_RESULTS_PATH
from highjump_mlops.features.engineering import build_features


def main() -> None:
    if not RAW_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Raw result file not found: {RAW_RESULTS_PATH}. "
            "Fetch the raw data inside Docker first:\n\n"
            "make fetch"
        )

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = pd.read_parquet(RAW_RESULTS_PATH)
    features = build_features(results)

    features.to_parquet(FEATURES_PATH, index=False)

    print(f"Saved {len(features)} feature rows to {FEATURES_PATH}", flush=True)
    print(features.head())


if __name__ == "__main__":
    main()
