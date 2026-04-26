import pandas as pd

from highjump_mlops.config import FEATURES_PATH, RAW_DIR, YEARS
from highjump_mlops.data_source import fetch_html, find_last_page, parse_toplist
from highjump_mlops.features import build_features


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for year in YEARS:
        print(f"Fetching year: {year}")
        first_html = fetch_html(year, page=1)
        last_page = find_last_page(first_html)
        year_row_count = 0

        for page in range(1, last_page + 1):
            html = first_html if page == 1 else fetch_html(year, page)
            (RAW_DIR / f"world_athletics_toplist_{year}_page_{page}.html").write_text(html)

            page_results = parse_toplist(html, year)

            if page_results.empty:
                break

            all_results.append(page_results)
            year_row_count += len(page_results)

        print(f"{year}: fetched {year_row_count} rows from {last_page} pages")

    results = pd.concat(all_results, ignore_index=True)
    features = build_features(results)

    features.to_parquet(FEATURES_PATH, index=False)

    print(f"Saved {len(features)} feature rows to {FEATURES_PATH}")
    print(features.head())


if __name__ == "__main__":
    main()
