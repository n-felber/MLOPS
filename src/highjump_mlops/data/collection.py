import logging
import random
import time
from pathlib import Path

import pandas as pd
import requests

from highjump_mlops.config import CURRENT_YEAR, RAW_DIR, RAW_RESULTS_PATH, YEARS
from highjump_mlops.data.source import fetch_html, find_last_page, parse_toplist


PAGE_RETRIES = 4
FETCH_LOG_PATH = RAW_DIR / "fetch.log"


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=FETCH_LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def raw_html_path(year: int, page: int) -> Path:
    return RAW_DIR / f"world_athletics_toplist_{year}_page_{page}.html"


def should_refresh_year(year: int) -> bool:
    return year == CURRENT_YEAR


def render_year_progress(year: int, current_page: int, total_pages: int, row_count: int, skipped_pages: int) -> None:
    bar_width = 30
    filled_width = round(bar_width * current_page / total_pages)
    bar = "#" * filled_width + "-" * (bar_width - filled_width)

    print(
        f"\r{year}: [{bar}] {current_page}/{total_pages} pages "
        f"| rows: {row_count} | skipped: {skipped_pages}",
        end="",
        flush=True,
    )


def wait_before_retry(attempt: int) -> None:
    min_seconds = (attempt + 1) ** 2
    max_seconds = (attempt + 2) ** 2
    delay = random.uniform(min_seconds, max_seconds)

    logger.info("Waiting %.1f seconds before retry", delay)
    time.sleep(delay)


def load_or_fetch_html(year: int, page: int, force_fetch: bool = False) -> tuple[str, bool]:
    path = raw_html_path(year, page)

    if path.exists() and not force_fetch:
        logger.info("Using cached HTML for %s page %s", year, page)
        return path.read_text(encoding="utf-8"), True

    logger.info("Fetching %s page %s", year, page)
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

        try:
            html, from_cache = load_or_fetch_html(
                year,
                page,
                force_fetch=force_fetch,
            )
        except requests.RequestException as error:
            logger.warning("Request failed for %s page %s: %s", year, page, error)

            if attempt < PAGE_RETRIES:
                wait_before_retry(attempt)
                continue

            break

        page_results = parse_toplist(html, year, page)

        if not page_results.empty:
            return html, page_results

        if attempt < PAGE_RETRIES:
            if from_cache:
                logger.info(
                    "Cached HTML for %s page %s did not contain usable data. "
                    "Refetching next attempt.",
                    year,
                    page,
                )
            else:
                logger.info(
                    "Retrying %s page %s: attempt %s/%s",
                    year,
                    page,
                    attempt,
                    PAGE_RETRIES,
                )
                wait_before_retry(attempt)

    logger.warning("Skipped %s page %s after %s failed attempts", year, page, PAGE_RETRIES)
    return "", pd.DataFrame()


def collect_year(year: int) -> pd.DataFrame:
    force_refresh = should_refresh_year(year)
    mode = "refreshing current year" if force_refresh else "using cache when valid"

    logger.info("Processing year %s (%s)", year, mode)

    first_html, first_page_results = load_valid_page(
        year,
        page=1,
        force_refresh=force_refresh,
    )

    if first_page_results.empty:
        print(f"{year}: skipped, page 1 did not contain usable data", flush=True)
        logger.warning("Skipping %s: page 1 did not contain usable data", year)
        return pd.DataFrame()

    last_page = find_last_page(first_html)

    year_results = [first_page_results]
    year_row_count = len(first_page_results)
    skipped_pages = 0

    render_year_progress(year, 1, last_page, year_row_count, skipped_pages)

    for page in range(2, last_page + 1):
        _, page_results = load_valid_page(
            year,
            page=page,
            force_refresh=force_refresh,
        )

        if page_results.empty:
            skipped_pages += 1
        else:
            year_results.append(page_results)
            year_row_count += len(page_results)

        render_year_progress(year, page, last_page, year_row_count, skipped_pages)

    print()
    logger.info(
        "%s: collected %s rows from %s pages (%s skipped)",
        year,
        year_row_count,
        last_page,
        skipped_pages,
    )

    return pd.concat(year_results, ignore_index=True)


def main() -> None:
    setup_logging()

    all_results = []

    print(f"Collecting years: {YEARS}", flush=True)
    print(f"Detailed fetch log: {FETCH_LOG_PATH}", flush=True)
    logger.info("Collecting years: %s", YEARS)

    for year in YEARS:
        year_results = collect_year(year)

        if year_results.empty:
            continue

        all_results.append(year_results)

    if not all_results:
        raise ValueError("No results were collected. Cannot save raw results.")

    results = pd.concat(all_results, ignore_index=True)

    results.to_parquet(RAW_RESULTS_PATH, index=False)

    print(f"Saved {len(results)} raw result rows to {RAW_RESULTS_PATH}", flush=True)
    logger.info("Saved %s raw result rows to %s", len(results), RAW_RESULTS_PATH)


if __name__ == "__main__":
    main()
