import pandas as pd

from highjump_mlops.data.source import find_last_page, parse_toplist


def test_find_last_page_uses_data_max_when_available() -> None:
    html = """
    <html>
        <button data-max="1">1</button>
        <button data-max="7">7</button>
        <button data-page="3">3</button>
    </html>
    """

    assert find_last_page(html) == 7


def test_find_last_page_falls_back_to_data_page() -> None:
    html = """
    <html>
        <button data-page="1">1</button>
        <button data-page="4">4</button>
    </html>
    """

    assert find_last_page(html) == 4


def test_parse_toplist_extracts_expected_columns_and_types() -> None:
    html = """
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Mark</th>
                <th>Competitor</th>
                <th>DOB</th>
                <th>Pos</th>
                <th>Venue</th>
                <th>Date</th>
                <th>Results Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td>2.30</td>
                <td> Test ATHLETE </td>
                <td>01 JAN 2000</td>
                <td>1</td>
                <td>Test Venue</td>
                <td>01 Jan 2024</td>
                <td>1200</td>
            </tr>
            <tr>
                <td>2</td>
                <td>2.25</td>
                <td>Other ATHLETE</td>
                <td>02 FEB 2001</td>
                <td>2</td>
                <td>Other Venue</td>
                <td>02 Jan 2024</td>
                <td>1150</td>
            </tr>
        </tbody>
    </table>
    """

    result = parse_toplist(html, year=2024, page=3)

    assert len(result) == 2
    assert list(result.columns) == [
        "rank",
        "mark",
        "competitor",
        "dob",
        "pos",
        "venue",
        "date",
        "results_score",
        "year",
        "source_page",
    ]

    first_row = result.iloc[0]

    assert first_row["rank"] == 1
    assert first_row["mark"] == 2.30
    assert first_row["competitor"] == "Test ATHLETE"
    assert first_row["date"] == pd.Timestamp("2024-01-01")
    assert first_row["results_score"] == 1200
    assert first_row["year"] == 2024
    assert first_row["source_page"] == 3


def test_parse_toplist_returns_empty_dataframe_for_missing_columns() -> None:
    html = """
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Mark</th>
                <th>Competitor</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td>2.30</td>
                <td>Test ATHLETE</td>
            </tr>
        </tbody>
    </table>
    """

    result = parse_toplist(html, year=2024, page=1)

    assert result.empty
