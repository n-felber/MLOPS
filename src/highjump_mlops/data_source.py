import re
from io import StringIO

import pandas as pd
import requests

from highjump_mlops.config import toplist_url


def fetch_html(year: int, page: int) -> str:
    response = requests.get(
        toplist_url(year, page),
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def find_last_page(html: str) -> int:
    data_max_values = re.findall(r'data-max="(\d+)"', html)

    if data_max_values:
        return max(int(value) for value in data_max_values)

    data_page_values = re.findall(r'data-page="(\d+)"', html)

    if data_page_values:
        return max(int(value) for value in data_page_values)

    return 1


def parse_toplist(html: str, year: int) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))

    if not tables:
        raise RuntimeError(f"No tables found for {year}.")

    df = tables[0]
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]

    df = df[["rank", "mark", "competitor", "date", "results_score"]].copy()
    df["year"] = year

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["mark"] = pd.to_numeric(df["mark"], errors="coerce")
    df["results_score"] = pd.to_numeric(df["results_score"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")

    return df.dropna(subset=["rank", "mark", "competitor", "date", "results_score"])
