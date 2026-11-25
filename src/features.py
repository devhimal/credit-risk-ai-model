"""Reusable feature engineering, preprocessing, and data splitting logic.

The :class:`FeatureEngineer` transformer is the *single* place where raw
applicant fields are converted into model inputs. It is embedded inside the
scikit-learn pipeline that is persisted with Joblib, so the API and Streamlit
app never re-implement preprocessing.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src import config

RAW_REQUIRED_FEATURES = list(config.RAW_FEATURES)

RATIO_FEATURES = [
    "utilization_ratio",
    "payment_ratio",
    "recent_repayment_ratio",
    "payments_to_limit",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived, leakage-free credit features.

    All inputs describe the six months *before* the target month. No target
    values, future statements, or row identifiers are used.
    """
    frame = df.copy()
    for column in config.BASE_NUMERIC_FEATURES:
        if column not in frame.columns:
            raise ValueError(f"Missing required feature: {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    bills = frame[config.BILL_COLUMNS]
    payments = frame[config.PAY_AMT_COLUMNS]
    status = frame[config.PAY_COLUMNS]

    frame["avg_bill_amt"] = bills.mean(axis=1)
    frame["total_bill_amt"] = bills.sum(axis=1)
    frame["avg_pay_amt"] = payments.mean(axis=1)
    frame["total_pay_amt"] = payments.sum(axis=1)

    limit_denominator = frame["limit_bal"].abs() + 1.0
    frame["utilization_ratio"] = frame["avg_bill_amt"] / limit_denominator
    frame["payment_ratio"] = frame["total_pay_amt"] / (
        frame["total_bill_amt"].abs() + 1.0
    )
    frame["recent_repayment_ratio"] = frame["pay_amt1"] / (
        frame["bill_amt1"].abs() + 1.0
    )
    frame["payments_to_limit"] = frame["total_pay_amt"] / limit_denominator

    frame["max_delinquency"] = status.max(axis=1)
    frame["num_delinquent_months"] = (status > 0).sum(axis=1)
    frame["delinquency_severity"] = status.clip(lower=0).sum(axis=1)
    frame["avg_delinquency"] = status.mean(axis=1)
    frame["balance_trend"] = frame["bill_amt1"] - frame["bill_amt6"]

    # Guard against division artefacts and absurd ratios without discarding
    # legitimate negative balances.
    for column in RATIO_FEATURES:
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan).clip(-20, 20)
    frame["balance_trend"] = (
        frame["balance_trend"].replace([np.inf, -np.inf], np.nan).clip(-1e9, 1e9)
    )
    return frame


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the modelling frame from raw applicant data.

    Parameters
    ----------
    clip_ratio:
        Absolute cap applied to ratio features to keep them numerically stable.
    """

    def __init__(self, clip_ratio: float = 20.0):
        self.clip_ratio = clip_ratio

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        self._validate(X)
        self.n_features_in_ = len(RAW_REQUIRED_FEATURES)
        self.feature_names_in_ = np.array(RAW_REQUIRED_FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        frame = add_engineered_features(pd.DataFrame(X))
        for column in RATIO_FEATURES:
            frame[column] = frame[column].clip(-self.clip_ratio, self.clip_ratio)
        ordered = [*config.NUMERIC_FEATURES, *config.CATEGORICAL_FEATURES]
        return frame[ordered].replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        if not isinstance(X, (pd.DataFrame, dict)):
            raise TypeError(
                "FeatureEngineer expects a pandas DataFrame or a dict of columns"
            )
        frame = pd.DataFrame(X)
        missing = sorted(set(RAW_REQUIRED_FEATURES) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing transformer (imputation, scaling, encoding)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(classifier) -> Pipeline:
    """Build the full, serialisable preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 70/15/15 train/validation/test split.

    The validation set is used only for threshold selection and model
    comparison; the test set is touched once for the final model.
    """
    X = df[RAW_REQUIRED_FEATURES].copy()
    y = df[config.TARGET].astype(int).copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=config.VAL_SIZE + config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    relative_test = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=config.RANDOM_SEED,
    )
    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def clean_feature_name(name: str) -> str:
    """Turn a ColumnTransformer output name into a readable label."""
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Return output feature names after preprocessing has been fitted."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:  # pragma: no cover - fallback for older sklearn
        return []
    return [clean_feature_name(str(name)) for name in names]


def coerce_input_frame(payload: dict | Iterable[dict]) -> pd.DataFrame:
    """Build a validated DataFrame from API/UI payload(s)."""
    records = [payload] if isinstance(payload, dict) else list(payload)
    frame = pd.DataFrame(records)
    missing = sorted(set(RAW_REQUIRED_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    return frame[RAW_REQUIRED_FEATURES]
