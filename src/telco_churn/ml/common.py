"""Shared data-loading, feature, preprocessing, and split utilities."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn.config import DB_PATH, TRAINING_SQL_PATH

TARGET_COLUMN = "churn_flag"
ID_COLUMN = "customer_id"

# total_charges is deliberately excluded from the primary model because it is
# cumulative and tied to each customer's observed subscription duration.
# tenure is retained for retrospective classification, but prevents claims of
# a validated forward-looking model without dated point-in-time snapshots.
NUMERIC_FEATURES = ["tenure_months", "monthly_charges"]
CATEGORICAL_FEATURES = [
    "gender",
    "is_senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_type",
    "payment_method",
    "paperless_billing",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_modeling_data() -> pd.DataFrame:
    """Load the one-row-per-customer modeling dataset from SQLite."""
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_PATH}. Run the ingestion and dbt steps first."
        )
    if not TRAINING_SQL_PATH.is_file():
        raise FileNotFoundError(f"Training SQL file not found: {TRAINING_SQL_PATH}")

    query = TRAINING_SQL_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(query, conn)

    required = {ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Modeling dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("The modeling query returned no rows.")
    if df[ID_COLUMN].isna().any():
        raise ValueError("customer_id contains missing values.")
    if df[ID_COLUMN].duplicated().any():
        duplicate_count = int(df[ID_COLUMN].duplicated().sum())
        raise ValueError(
            f"The modeling query returned duplicate customer IDs ({duplicate_count}). "
            "Check dimension key uniqueness and fact-to-dimension joins."
        )

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("churn_flag must contain only non-null binary values 0 and 1.")
    df[TARGET_COLUMN] = target.astype("int8")

    for col in NUMERIC_FEATURES:
        raw = df[col].astype("string").str.strip()

        blank = raw.isna() | raw.eq("")
        converted = pd.to_numeric(
            raw.mask(blank),
            errors="coerce",
        )

        invalid = (~blank) & converted.isna()

        if invalid.any():
            raise ValueError(
                f"Column '{col}' contains invalid numeric values."
            )

        if (converted.dropna() < 0).any():
            raise ValueError(
                f"Column '{col}' contains negative values."
            )

        df[col] = converted

    # Do not convert missing categorical values into literal strings such as 'nan'.
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fitted only on each training fold/split."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_train_validation_test_split(
    df: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Any]:
    """Create stratified 70%/15%/15% train/validation/test partitions."""
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
