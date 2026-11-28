"""Data manifests: where a dataset comes from, so that data need not be stored with a model.

A manifest is a small YAML file:

    source:
      url: https://.../replication-files.zip
      sha256: <checksum of the downloaded file>
      member: data.xlsx          # the file inside a zip; omit for a plain file
      sheet: Data                # for a spreadsheet; omit for CSV
      date_column: Date
      first_date: 1983Q1         # for a CSV file with no date column: its rows are consecutive quarters from this one
      date_column: ""            # (a date column with an empty heading)
    sample: {first: 1984Q1, last: 2019Q4}
    demean: true                 # optional: subtract each series' mean over the sample (Dynare's prefilter=1)
    attribution: who made the data, and on what terms

`load_manifest` downloads the file once into a cache outside the repository (~/.cache/mutatis, or $MUTATIS_CACHE), checks the
checksum, and returns the dataset.
"""

import hashlib
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import yaml

from mutatis.measure.data import DataError, Dataset


def cache_folder() -> Path:
    return Path(os.environ.get("MUTATIS_CACHE", Path.home() / ".cache" / "mutatis"))


def fetch(url: str, sha256: str) -> Path:
    target = cache_folder() / sha256
    if target.exists():
        return target
    request = urllib.request.Request(url, headers={"User-Agent": "mutatis"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    found = hashlib.sha256(content).hexdigest()
    if found != sha256:
        raise DataError(f"the file at {url} has checksum {found}, not the {sha256} the manifest expects")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def load_manifest(path: str | Path) -> Dataset:
    manifest = yaml.safe_load(Path(path).read_text())
    source = manifest["source"]
    if not isinstance(source.get("sha256"), str):
        raise DataError("put the sha256 checksum in quotes: YAML reads a checksum made only of digits as a number and drops its leading zeros")
    downloaded = fetch(source["url"], source["sha256"])
    if "member" in source:
        with zipfile.ZipFile(downloaded) as archive:
            raw = archive.read(source["member"])
        name = source["member"]
    else:
        raw, name = downloaded.read_bytes(), source["url"]
    if name.lower().endswith((".xlsx", ".xlsm")):
        dataset = Dataset.read_excel(io.BytesIO(raw), source["sheet"], source.get("date_column", "date"))
    else:
        scratch = cache_folder() / (hashlib.sha256(raw).hexdigest() + ".csv")
        scratch.write_bytes(raw)
        dataset = Dataset.read_csv(scratch, source.get("first_date"))
    sample = manifest.get("sample", {})
    dataset = dataset.between(sample.get("first"), sample.get("last"))
    return dataset.demeaned() if manifest.get("demean", False) else dataset
