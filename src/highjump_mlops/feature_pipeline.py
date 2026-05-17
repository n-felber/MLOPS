import random
import time
import pandas as pd

from highjump_mlops.config import FEATURES_PATH, RAW_DIR, YEARS
from highjump_mlops.data_source import fetch_html, find_last_page, parse_toplist
from highjump_mlops.features import build_features


PAGE_RETRIES = 4


def wait_before_retry(attempt: int) -> None:
    min_seconds = (attempt + 1) ** 2
    max_seconds = (attempt + 2) ** 2
    delay = random.uniform(min_seconds, max_seconds)

    print(f"Waiting {delay:.1f} seconds before retry", flush=True)
    time.sleep(delay)


def fetch_and_parse_page(year: int, page: int) -> pd.DataFrame:
    for attempt in range(1, PAGE_RETRIES + 1):
        html = fetch_html(year, page)
        page_results = parse_toplist(html, year, page)

        if not page_results.empty:
            (RAW_DIR / f"world_athletics_toplist_{year}_page_{page}.html").write_text(html)
            return page_results

        if attempt < PAGE_RETRIES:
            print(
                f"Retrying {year} page {page}: attempt {attempt}/{PAGE_RETRIES}",
                flush=True,
            )
            wait_before_retry(attempt)

    print(f"Skipped {year} page {page} after {PAGE_RETRIES} failed attempts", flush=True)
    return pd.DataFrame()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for year in YEARS:
        print(f"Fetching year: {year}", flush=True)

        first_html = fetch_html(year, page=1)
        last_page = find_last_page(first_html)

        year_row_count = 0
        skipped_pages = 0

        for page in range(1, last_page + 1):
            page_results = fetch_and_parse_page(year, page)

            if page_results.empty:
                skipped_pages += 1
                continue

            all_results.append(page_results)
            year_row_count += len(page_results)

        print(
            f"{year}: fetched {year_row_count} rows from {last_page} pages "
            f"({skipped_pages} skipped)",
            flush=True,
        )

    results = pd.concat(all_results, ignore_index=True)
    features = build_features(results)

    features.to_parquet(FEATURES_PATH, index=False)

    print(f"Saved {len(features)} feature rows to {FEATURES_PATH}", flush=True)
    print(features.head())


if __name__ == "__main__":
    main()
