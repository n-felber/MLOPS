import random
import re
import time
from io import StringIO

import pandas as pd
import requests

from highjump_mlops.config import toplist_url


REQUEST_DELAY_SECONDS = (0.7, 2.4)


def wait_between_requests() -> None:
    time.sleep(random.uniform(*REQUEST_DELAY_SECONDS))


def fetch_html(year: int, page: int) -> str:
    response = requests.get(
        toplist_url(year, page),
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()

    wait_between_requests()

    return response.text


def find_last_page(html: str) -> int:
    data_max_values = re.findall(r'data-max="(\d+)"', html)

    if data_max_values:
        return max(int(value) for value in data_max_values)

    data_page_values = re.findall(r'data-page="(\d+)"', html)

    if data_page_values:
        return max(int(value) for value in data_page_values)

    return 1


def parse_toplist(html: str, year: int, page: int) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, ImportError) as error:
        print(f"Could not parse table for {year} page {page}: {error}", flush=True)
        return pd.DataFrame()

    if not tables:
        print(f"No tables found for {year} page {page}", flush=True)
        return pd.DataFrame()

    df = tables[0]
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]

    expected_columns = [
        "rank",
        "mark",
        "competitor",
        "dob",
        "pos",
        "venue",
        "date",
        "results_score",
    ]

    missing_columns = [column for column in expected_columns if column not in df.columns]

    if missing_columns:
        print(
            f"Missing expected columns for {year} page {page}: {missing_columns}",
            flush=True,
        )
        print(f"Available columns: {list(df.columns)}", flush=True)
        return pd.DataFrame()

    df = df[expected_columns].copy()
    df["year"] = year
    df["source_page"] = page

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["mark"] = pd.to_numeric(df["mark"], errors="coerce")
    df["results_score"] = pd.to_numeric(df["results_score"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")

    df["competitor"] = df["competitor"].astype(str).str.strip()
    df["dob"] = df["dob"].astype(str).str.strip()
    df["pos"] = df["pos"].astype(str).str.strip()
    df["venue"] = df["venue"].astype(str).str.strip()

    return df.dropna(
        subset=[
            "rank",
            "mark",
            "competitor",
            "date",
            "results_score",
        ]
    )