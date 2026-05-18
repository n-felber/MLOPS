import random
import time
from pathlib import Path

import pandas as pd

from highjump_mlops.config import CURRENT_YEAR, FEATURES_PATH, RAW_DIR, YEARS
from highjump_mlops.data.source import fetch_html, find_last_page, parse_toplist
from highjump_mlops.features.engineering import build_features


PAGE_RETRIES = 4


def raw_html_path(year: int, page: int) -> Path:
    return RAW_DIR / f"world_athletics_toplist_{year}_page_{page}.html"


def should_refresh_year(year: int) -> bool:
    return year == CURRENT_YEAR


def wait_before_retry(attempt: int) -> None:
    min_seconds = (attempt + 1) ** 2
    max_seconds = (attempt + 2) ** 2
    delay = random.uniform(min_seconds, max_seconds)

    print(f"Waiting {delay:.1f} seconds before retry", flush=True)
    time.sleep(delay)


def load_or_fetch_html(year: int, page: int, force_fetch: bool = False) -> tuple[str, bool]:
    path = raw_html_path(year, page)

    if path.exists() and not force_fetch:
        print(f"Using cached HTML for {year} page {page}", flush=True)
        return path.read_text(encoding="utf-8"), True

    print(f"Fetching {year} page {page}", flush=True)
    html = fetch_html(year, page)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")

    return html, False


def load_valid_page(
    year: int,
    page: int,
    force_refresh: bool = False,
) -> tuple[str, pd.DataFrame]:
    for attempt in range(1, PAGE_RETRIES + 1):
        force_fetch = force_refresh or attempt > 1

        html, from_cache = load_or_fetch_html(
            year,
            page,
            force_fetch=force_fetch,
        )

        page_results = parse_toplist(html, year, page)

        if not page_results.empty:
            return html, page_results

        if attempt < PAGE_RETRIES:
            if from_cache:
                print(
                    f"Cached HTML for {year} page {page} did not contain usable data. "
                    "Refetching next attempt.",
                    flush=True,
                )
            else:
                print(
                    f"Retrying {year} page {page}: attempt {attempt}/{PAGE_RETRIES}",
                    flush=True,
                )
                wait_before_retry(attempt)

    print(f"Skipped {year} page {page} after {PAGE_RETRIES} failed attempts", flush=True)
    return "", pd.DataFrame()


def collect_year(year: int) -> pd.DataFrame:
    force_refresh = should_refresh_year(year)

    if force_refresh:
        print(f"Processing year: {year} (refreshing current year)", flush=True)
    else:
        print(f"Processing year: {year} (using cache when valid)", flush=True)

    first_html, first_page_results = load_valid_page(
        year,
        page=1,
        force_refresh=force_refresh,
    )

    if first_page_results.empty:
        print(f"Skipping {year}: page 1 did not contain usable data", flush=True)
        return pd.DataFrame()

    last_page = find_last_page(first_html)

    year_results = [first_page_results]
    year_row_count = len(first_page_results)
    skipped_pages = 0

    for page in range(2, last_page + 1):
        _, page_results = load_valid_page(
            year,
            page=page,
            force_refresh=force_refresh,
        )

        if page_results.empty:
            skipped_pages += 1
            continue

        year_results.append(page_results)
        year_row_count += len(page_results)

    print(
        f"{year}: collected {year_row_count} rows from {last_page} pages "
        f"({skipped_pages} skipped)",
        flush=True,
    )

    return pd.concat(year_results, ignore_index=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"Collecting years: {YEARS}", flush=True)

    for year in YEARS:
        year_results = collect_year(year)

        if year_results.empty:
            continue

        all_results.append(year_results)

    if not all_results:
        raise ValueError("No results were collected. Cannot build features.")

    results = pd.concat(all_results, ignore_index=True)
    features = build_features(results)

    features.to_parquet(FEATURES_PATH, index=False)

    print(f"Saved {len(features)} feature rows to {FEATURES_PATH}", flush=True)
    print(features.head())


if __name__ == "__main__":
    main()
