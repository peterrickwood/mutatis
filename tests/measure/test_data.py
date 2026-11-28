"""Datasets, transforms, calendar variables, spreadsheets, manifests, and reading the starting state from data."""

import hashlib
import zipfile

import numpy as np
import pytest

from mutatis.measure import manifest
from mutatis.measure.data import DataError, Dataset, calendar_values, quarters, transform_series


def test_transforms_are_expressions_in_x():
    x = np.array([100.0, 110.0, 121.0])
    np.testing.assert_allclose(transform_series("100 * log(x)", x), 100 * np.log(x))
    np.testing.assert_allclose(transform_series("d(log(x))", x), [np.nan, np.log(1.1), np.log(1.1)])
    np.testing.assert_allclose(transform_series("x / x(-1) - 1", x), [np.nan, 0.1, 0.1])
    np.testing.assert_array_equal(transform_series(None, x), x)
    with pytest.raises(DataError):
        transform_series("x(+1)", x)
    with pytest.raises(DataError):
        transform_series("y * 2", x)


def test_quarters_and_calendar():
    dates = quarters("2007Q3", 4)
    assert dates == ("2007Q3", "2007Q4", "2008Q1", "2008Q2")
    np.testing.assert_array_equal(calendar_values("date < 2008Q2", dates), [1, 1, 1, 0])


def test_csv_round_trip_and_sample(tmp_path):
    data = Dataset(quarters("2000Q1", 5), {"a": np.arange(5.0), "b": np.array([1.0, np.nan, 3.0, 4.0, 5.0])})
    data.write_csv(tmp_path / "d.csv")
    back = Dataset.read_csv(tmp_path / "d.csv")
    assert back.dates == data.dates
    np.testing.assert_array_equal(back.series["b"], data.series["b"])
    middle = back.between("2000Q2", "2000Q4")
    assert middle.dates == ("2000Q2", "2000Q3", "2000Q4") and list(middle.series["a"]) == [1.0, 2.0, 3.0]


def _spreadsheet(path):
    import datetime

    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Data"
    sheet.append(["Date", "GDP", "RATE"])
    for i, month in enumerate([3, 6, 9, 12]):
        sheet.append([datetime.datetime(1999, month, 30 if month in (6, 9) else 31), 100.0 + i, None if i == 2 else 5.0])
    book.save(path)


def test_spreadsheet_dates_become_quarters(tmp_path):
    _spreadsheet(tmp_path / "data.xlsx")
    data = Dataset.read_excel(tmp_path / "data.xlsx", "Data", "Date")
    assert data.dates == ("1999Q1", "1999Q2", "1999Q3", "1999Q4")
    assert np.isnan(data.series["RATE"][2]) and data.series["GDP"][3] == 103.0


def test_manifest_downloads_once_checks_the_checksum_and_selects_the_sample(tmp_path, monkeypatch):
    _spreadsheet(tmp_path / "data.xlsx")
    with zipfile.ZipFile(tmp_path / "files.zip", "w") as archive:
        archive.write(tmp_path / "data.xlsx", "data.xlsx")
    checksum = hashlib.sha256((tmp_path / "files.zip").read_bytes()).hexdigest()
    monkeypatch.setenv("MUTATIS_CACHE", str(tmp_path / "cache"))
    (tmp_path / "manifest.yaml").write_text(f"""
source:
  url: {(tmp_path / 'files.zip').as_uri()}
  sha256: {checksum}
  member: data.xlsx
  sheet: Data
  date_column: Date
sample: {{first: 1999Q2, last: 1999Q3}}
""")
    data = manifest.load_manifest(tmp_path / "manifest.yaml")
    assert data.dates == ("1999Q2", "1999Q3")
    assert (tmp_path / "cache" / checksum).exists()
    (tmp_path / "manifest.yaml").write_text((tmp_path / "manifest.yaml").read_text().replace(checksum, "0" * 64))
    with pytest.raises(DataError, match="in quotes"):  # all digits: YAML would read it as a number
        manifest.load_manifest(tmp_path / "manifest.yaml")
    (tmp_path / "manifest.yaml").write_text((tmp_path / "manifest.yaml").read_text().replace("0" * 64, '"' + "1" * 63 + 'a"'))
    with pytest.raises(DataError, match="checksum"):
        manifest.load_manifest(tmp_path / "manifest.yaml")


def test_an_unnamed_date_column_and_demeaning_over_the_sample(tmp_path, monkeypatch):
    """Dynare's data sheets often leave the date column's heading empty; `prefilter=1` demeans each series over the estimation
    sample, using the values present."""
    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Input"
    for row in [(None, "a", "b"), ("1999Q1", 1.0, "#N/A"), ("1999Q2", 2.0, 4.0), ("1999Q3", 3.0, 8.0), ("1999Q4", 10.0, 9.0)]:
        sheet.append(row)
    book.save(tmp_path / "data.xlsx")
    data = Dataset.read_excel(tmp_path / "data.xlsx", "Input", "")
    assert data.dates[0] == "1999Q1" and np.isnan(data.series["b"][0])
    checksum = hashlib.sha256((tmp_path / "data.xlsx").read_bytes()).hexdigest()
    monkeypatch.setenv("MUTATIS_CACHE", str(tmp_path / "cache"))
    (tmp_path / "manifest.yaml").write_text(f"""
source: {{url: {(tmp_path / 'data.xlsx').as_uri()}, sha256: "{checksum}", sheet: Input, date_column: ""}}
sample: {{first: 1999Q1, last: 1999Q3}}
demean: true
""")
    data = manifest.load_manifest(tmp_path / "manifest.yaml")
    np.testing.assert_allclose(data.series["a"], [-1.0, 0.0, 1.0])
    assert np.isnan(data.series["b"][0])
    np.testing.assert_allclose(data.series["b"][1:], [-2.0, 2.0])

