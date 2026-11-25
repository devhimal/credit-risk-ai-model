"""Central configuration for the credit-default risk project.

All paths are resolved relative to the repository root so that scripts,
notebooks, the API, and the Streamlit app behave identically regardless of
the current working directory.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLFLOW_ARTIFACTS_DIR = ROOT_DIR / "mlartifacts"
MLFLOW_DB = ROOT_DIR / "mlflow.db"

RAW_ZIP_PATH = RAW_DIR / "default_of_credit_card_clients.zip"
RAW_XLS_PATH = RAW_DIR / "default of credit card clients.xls"
RAW_CSV_PATH = RAW_DIR / "credit_card_default_raw.csv"
PROCESSED_CSV_PATH = PROCESSED_DIR / "credit_card_default_clean.csv"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_META_PATH = MODELS_DIR / "model.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.pdf"

# ---------------------------------------------------------------------------
# Dataset provenance (verified 2026-11-25 from UCI dataset page)
# ---------------------------------------------------------------------------
DATASET_NAME = "Default of Credit Card Clients"
DATASET_URN = "uci:350"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
DATASET_DOI = "10.24432/C55S3H"
DATASET_LICENSE = "CC BY 4.0 (Creative Commons Attribution 4.0 International)"
DATASET_CITATION = (
    "Yeh, I. (2009). Default of Credit Card Clients [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H"
)
# MD5 of the original UCI zip archive observed on 2026-11-25.
EXPECTED_ZIP_MD5 = "4fc9f7278533d4f40e0b9dcc1c0a40f8"

# ---------------------------------------------------------------------------
# Modelling configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TARGET = "default_next_month"
ID_COLUMN = "ID"

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Relative business costs used for threshold selection. A false negative
# (approving an account that defaults) is treated as 5x more expensive than a
# false positive (unnecessary manual review). These are *assumptions* and are
# sensitivity-tested in the evaluation notebook.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0
REVIEW_CAPACITY_FRACTION = 0.30  # ops can manually review up to 30% of accounts

DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = [round(0.02 + 0.01 * i, 2) for i in range(94)]  # 0.02..0.95

# Stable column names used by the trained pipeline and the API.
PAY_COLUMNS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]  # UCI has no PAY_1
BILL_COLUMNS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLUMNS = [f"pay_amt{i}" for i in range(1, 7)]
CATEGORICAL_FEATURES = ["sex", "education", "marriage"]
BASE_NUMERIC_FEATURES = ["limit_bal", "age", *PAY_COLUMNS, *BILL_COLUMNS, *PAY_AMT_COLUMNS]
ENGINEERED_FEATURES = [
    "avg_bill_amt",
    "total_bill_amt",
    "avg_pay_amt",
    "total_pay_amt",
    "utilization_ratio",
    "payment_ratio",
    "max_delinquency",
    "num_delinquent_months",
    "delinquency_severity",
    "recent_repayment_ratio",
    "balance_trend",
    "payments_to_limit",
    "avg_delinquency",
]
NUMERIC_FEATURES = [*BASE_NUMERIC_FEATURES, *ENGINEERED_FEATURES]
ALL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
RAW_FEATURES = [*BASE_NUMERIC_FEATURES, *CATEGORICAL_FEATURES]

# Mapping from the original UCI column names to the names used in this project.
RENAME_MAP = {
    "ID": "ID",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": TARGET,
}

# ---------------------------------------------------------------------------
# Experiment tracking / serving
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "credit-default-risk"
FINAL_MODEL_TAG = "final_model"
MODEL_VERSION = "1.0.0"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
API_HOST = "0.0.0.0"
API_PORT = 8000

# Human-readable descriptions for every input accepted by the API/UI.
FEATURE_DESCRIPTIONS = {
    "limit_bal": "Approved credit limit (NT dollars)",
    "sex": "1 = male, 2 = female",
    "education": "1 = graduate school, 2 = university, 3 = high school, 4 = others",
    "marriage": "1 = married, 2 = single, 3 = others",
    "age": "Age in years",
    "pay_0": "Repayment status in Sep 2005 (-1 = paid duly, 1..9 = months delayed)",
    "pay_2": "Repayment status in Aug 2005",
    "pay_3": "Repayment status in Jul 2005",
    "pay_4": "Repayment status in Jun 2005",
    "pay_5": "Repayment status in May 2005",
    "pay_6": "Repayment status in Apr 2005",
    "bill_amt1": "Bill statement amount in Sep 2005 (NT dollars)",
    "bill_amt2": "Bill statement amount in Aug 2005",
    "bill_amt3": "Bill statement amount in Jul 2005",
    "bill_amt4": "Bill statement amount in Jun 2005",
    "bill_amt5": "Bill statement amount in May 2005",
    "bill_amt6": "Bill statement amount in Apr 2005",
    "pay_amt1": "Amount paid in Sep 2005 (NT dollars)",
    "pay_amt2": "Amount paid in Aug 2005",
    "pay_amt3": "Amount paid in Jul 2005",
    "pay_amt4": "Amount paid in Jun 2005",
    "pay_amt5": "Amount paid in May 2005",
    "pay_amt6": "Amount paid in Apr 2005",
    "default_next_month": "1 = defaulted on the next month's payment, 0 = did not",
}


def ensure_dirs() -> None:
    """Create all runtime directories if they do not exist."""
    for directory in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MLFLOW_ARTIFACTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def library_versions() -> dict[str, str]:
    """Collect versions of libraries that affect model serialisation."""
    versions = {"python": platform.python_version()}
    for module_name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "mlflow",
        "joblib",
        "fastapi",
        "streamlit",
        "pydantic",
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:  # pragma: no cover - defensive
            versions[module_name] = "not-installed"
    return versions


def python_executable() -> str:
    return sys.executable
