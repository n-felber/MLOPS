import pandas as pd

from highjump_mlops.config import FEATURES_PATH, RAW_DIR, YEARS
from highjump_mlops.data_source import fetch_html, parse_toplist
from highjump_mlops.features import build_features


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
