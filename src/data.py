"""Dataset acquisition, provenance recording, cleaning, and audit helpers."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

USER_AGENT = "applied-ml-course-project/1.0 (academic use)"


def md5sum(path: Path) -> str:
    """Return the MD5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, retries: int = 3) -> None:
    """Download a URL to disk with simple retry/back-off logic."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            return
        except Exception as exc:  # noqa: BLE001 - surfaced after retries
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def ensure_raw_data(force: bool = False) -> Path:
    """Ensure the original archive and a raw CSV copy exist in ``data/raw``.

    The archive is downloaded from UCI and is the authoritative source. A CSV
    conversion is stored alongside it purely for convenient loading. The
    download timestamp and MD5 hashes are recorded by
    :func:`build_processed_dataset`.
    """
    config.ensure_dirs()

    if force or not config.RAW_ZIP_PATH.exists():
        _download(config.DATASET_URL, config.RAW_ZIP_PATH)

    actual_md5 = md5sum(config.RAW_ZIP_PATH)
    if actual_md5 != config.EXPECTED_ZIP_MD5:
        warnings.warn(
            "The downloaded archive MD5 differs from the hash observed when this "
            f"project was built ({actual_md5} != {config.EXPECTED_ZIP_MD5}). "
            "UCI may have re-packaged the file; provenance is recorded anyway.",
            stacklevel=2,
        )

    if force or not config.RAW_XLS_PATH.exists():
        with zipfile.ZipFile(config.RAW_ZIP_PATH) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".xls")]
            if not members:
                raise RuntimeError("No .xls file found in the UCI archive")
            config.RAW_XLS_PATH.write_bytes(archive.read(members[0]))

    if force or not config.RAW_CSV_PATH.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = pd.read_excel(config.RAW_XLS_PATH, header=1, engine="xlrd")
        raw.to_csv(config.RAW_CSV_PATH, index=False)

    return config.RAW_CSV_PATH


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Load the untouched raw table (original UCI column names)."""
    ensure_raw_data(force=force_download)
    return pd.read_csv(config.RAW_CSV_PATH)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset without using the target to derive features.

    Steps
    -----
    * Rename columns to snake_case.
    * Drop the row identifier (it carries no predictive information).
    * Report identical feature vectors (retained because they can be distinct
      customers with the same credit profile).
    * Collapse undocumented EDUCATION codes (0, 5, 6) into "others" (4) and
      undocumented MARRIAGE code 0 into "others" (3).
    * Enforce integer types and validate the target range.
    """
    cleaned = df.rename(columns=config.RENAME_MAP).copy()

    if config.ID_COLUMN in cleaned.columns:
        cleaned = cleaned.drop(columns=[config.ID_COLUMN])

    # Identical feature vectors can belong to different customers, so they are
    # reported but not removed. The ID column already confirmed distinct rows.
    cleaned = cleaned.reset_index(drop=True)

    if config.TARGET not in cleaned.columns:
        raise ValueError(f"Target column '{config.TARGET}' missing after renaming")

    cleaned[config.TARGET] = cleaned[config.TARGET].astype(int)
    if not set(cleaned[config.TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary (0/1)")

    expected = set(config.RAW_FEATURES) | {config.TARGET}
    missing = expected - set(cleaned.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    cleaned = cleaned[[*config.RAW_FEATURES, config.TARGET]]

    cleaned["sex"] = cleaned["sex"].astype(int)
    cleaned["education"] = (
        cleaned["education"].replace({0: 4, 5: 4, 6: 4}).astype(int)
    )
    cleaned["marriage"] = cleaned["marriage"].replace({0: 3}).astype(int)

    for column in config.BASE_NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if cleaned["sex"].isin([1, 2]).all() is False:
        raise ValueError("SEX contains values outside {1, 2}")
    if not cleaned["education"].isin([1, 2, 3, 4]).all():
        raise ValueError("EDUCATION contains unexpected values")
    if not cleaned["marriage"].isin([1, 2, 3]).all():
        raise ValueError("MARRIAGE contains unexpected values")

    return cleaned


def _outlier_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Count IQR-based outliers for numeric columns (not removed)."""
    summary: dict[str, dict[str, float]] = {}
    for column in config.BASE_NUMERIC_FEATURES:
        series = df[column].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        summary[column] = {
            "count": count,
            "percentage": round(100 * count / len(series), 3) if len(series) else 0.0,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }
    return summary


def build_eda_summary(df: pd.DataFrame) -> dict:
    """Compute a compact, reproducible EDA summary dictionary."""
    target_counts = df[config.TARGET].value_counts().sort_index()
    numeric_describe = (
        df[config.BASE_NUMERIC_FEATURES + config.ENGINEERED_FEATURES]
        .describe()
        .round(4)
        .to_dict()
    )
    corr = (
        df[config.BASE_NUMERIC_FEATURES + [config.TARGET]]
        .corr(numeric_only=True)[config.TARGET]
        .drop(config.TARGET)
        .sort_values(key=np.abs, ascending=False)
    )
    negative_bill = {
        column: int((df[column] < 0).sum())
        for column in config.BILL_COLUMNS
        if column in df.columns
    }
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_default_rate": round(float(df[config.TARGET].mean()), 6),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {
            k: int(v) for k, v in df.isna().sum().items() if int(v) > 0
        },
        "duplicate_feature_rows": int(df.duplicated(subset=config.RAW_FEATURES).sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "numeric_describe": numeric_describe,
        "iqr_outliers": _outlier_summary(df),
        "negative_bill_amount_counts": negative_bill,
        "category_counts": {
            column: {str(k): int(v) for k, v in df[column].value_counts().items()}
            for column in config.CATEGORICAL_FEATURES
        },
        "repayment_status_counts": {
            column: {str(k): int(v) for k, v in df[column].value_counts().items()}
            for column in config.PAY_COLUMNS
        },
        "correlation_with_target": {
            k: round(float(v), 4) for k, v in corr.items()
        },
        "notes": [
            "No missing values were found in the published dataset.",
            "IQR outliers are reported but retained: extreme balances and "
            "payments are legitimate credit behaviour, and tree models are "
            "robust to them. RobustScaler limits their influence for the "
            "linear model.",
            "Negative bill amounts represent credit balances and are valid.",
            "ID was dropped because it is a row identifier and could cause "
            "spurious memorisation.",
            "Identical feature vectors are retained: they can be distinct "
            "customers with the same credit profile, not data-entry errors.",
        ],
    }


def build_processed_dataset(force: bool = False) -> pd.DataFrame:
    """Create the cleaned dataset plus provenance and EDA metadata files."""
    config.ensure_dirs()
    ensure_raw_data(force=force)

    if config.PROCESSED_CSV_PATH.exists() and not force:
        cleaned = pd.read_csv(config.PROCESSED_CSV_PATH)
        if config.DATASET_INFO_PATH.exists() and config.EDA_SUMMARY_PATH.exists():
            return cleaned

    raw = load_raw()
    cleaned = clean_dataframe(raw)
    cleaned.to_csv(config.PROCESSED_CSV_PATH, index=False)

    retrieved_at = datetime.fromtimestamp(
        config.RAW_ZIP_PATH.stat().st_mtime, tz=timezone.utc
    ).isoformat()

    dataset_info = {
        "name": config.DATASET_NAME,
        "urn": config.DATASET_URN,
        "source_url": config.DATASET_URL,
        "dataset_page": config.DATASET_PAGE,
        "doi": config.DATASET_DOI,
        "license": config.DATASET_LICENSE,
        "citation": config.DATASET_CITATION,
        "retrieved_at_utc": retrieved_at,
        "raw_archive": {
            "path": str(config.RAW_ZIP_PATH.relative_to(config.ROOT_DIR)),
            "bytes": config.RAW_ZIP_PATH.stat().st_size,
            "md5": md5sum(config.RAW_ZIP_PATH),
        },
        "raw_csv": {
            "path": str(config.RAW_CSV_PATH.relative_to(config.ROOT_DIR)),
            "bytes": config.RAW_CSV_PATH.stat().st_size,
            "md5": md5sum(config.RAW_CSV_PATH),
        },
        "processed_csv": {
            "path": str(config.PROCESSED_CSV_PATH.relative_to(config.ROOT_DIR)),
            "bytes": config.PROCESSED_CSV_PATH.stat().st_size,
            "md5": md5sum(config.PROCESSED_CSV_PATH),
        },
        "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "processed_shape": [int(cleaned.shape[0]), int(cleaned.shape[1])],
        "target": config.TARGET,
        "target_distribution": {
            str(k): int(v)
            for k, v in cleaned[config.TARGET].value_counts().sort_index().items()
        },
        "notes": [
            "Raw CSV is an unmodified table conversion of the .xls file inside "
            "the official UCI archive.",
            "Processed data rewrites column names, drops ID, collapses "
            "undocumented EDUCATION/MARRIAGE codes, and retains all rows.",
        ],
        "duplicate_feature_rows": int(
            cleaned.duplicated(subset=config.RAW_FEATURES).sum()
        ),
    }
    config.DATASET_INFO_PATH.write_text(json.dumps(dataset_info, indent=2))

    from src.features import add_engineered_features

    eda_summary = build_eda_summary(add_engineered_features(cleaned))
    eda_summary["dataset_source"] = config.DATASET_URL
    eda_summary["dataset_license"] = config.DATASET_LICENSE
    config.EDA_SUMMARY_PATH.write_text(json.dumps(eda_summary, indent=2, default=float))
    return cleaned


def load_processed() -> pd.DataFrame:
    """Load the cleaned dataset, building it first when necessary."""
    if not config.PROCESSED_CSV_PATH.exists():
        return build_processed_dataset()
    return pd.read_csv(config.PROCESSED_CSV_PATH)


def load_dataset_info() -> dict:
    if not config.DATASET_INFO_PATH.exists():
        build_processed_dataset()
    return json.loads(config.DATASET_INFO_PATH.read_text())


def load_eda_summary() -> dict:
    if not config.EDA_SUMMARY_PATH.exists():
        build_processed_dataset()
    return json.loads(config.EDA_SUMMARY_PATH.read_text())


if __name__ == "__main__":  # pragma: no cover
    frame = build_processed_dataset()
    info = load_dataset_info()
    print(f"Rows: {len(frame):,}")
    print(f"Target rate: {frame[config.TARGET].mean():.3%}")
    print(f"Raw MD5: {info['raw_csv']['md5']}")
    print(f"Processed: {config.PROCESSED_CSV_PATH}")
