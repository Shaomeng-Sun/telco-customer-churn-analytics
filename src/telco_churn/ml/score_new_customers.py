"""Score a separate, new customer snapshot supplied as a CSV file.

The CSV must use the feature column names produced by customer_training_dataset.sql.
Do not use the original training snapshot as evidence of prospective performance.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from telco_churn.config import METADATA_PATH, MODEL_PATH, resolve_project_path
from telco_churn.ml.common import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a separate customer feature CSV.")
    parser.add_argument("--input", required=True, help="CSV containing new customer features.")
    parser.add_argument(
        "--data",
        default="data/ml/new_customer_scores.csv",
        help="Output CSV path (relative paths are resolved from the project root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.data)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("Run train.py first to create the model and metadata.")

    new_data = pd.read_csv(input_path)

    # Reject empty input files.
    if new_data.empty:
        raise ValueError("Input CSV contains no customer records.")

    # Do not allow the target variable in the scoring input.
    if TARGET_COLUMN in new_data.columns:
        raise ValueError(
            f"Input CSV should not contain {TARGET_COLUMN}. "
            "Scoring does not require the actual churn outcome."
        )

    # Check that all model features are present.
    missing = sorted(set(FEATURE_COLUMNS).difference(new_data.columns))

    if missing:
        raise ValueError(
            f"Input CSV is missing required feature columns: {missing}"
        )

    # Validate customer IDs if supplied.
    if ID_COLUMN in new_data.columns:
        ids = new_data[ID_COLUMN].astype("string").str.strip()

        if ids.isna().any() or ids.eq("").any():
            raise ValueError("Input CSV contains missing or blank customer IDs.")

        if ids.duplicated().any():
            raise ValueError("Input CSV contains duplicate customer IDs.")

        new_data[ID_COLUMN] = ids

    # Convert numeric columns and identify invalid values.
    for col in NUMERIC_FEATURES:
        original = new_data[col].replace(r"^\s*$", np.nan, regex=True)
        converted = pd.to_numeric(original, errors="coerce")

        invalid = original.notna() & converted.isna()

        if invalid.any():
            raise ValueError(
                f"Column '{col}' contains non-numeric values."
            )

        new_data[col] = converted

    model = joblib.load(MODEL_PATH)
    metadata = load_json(METADATA_PATH)

    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError(
            "Input feature definitions do not match the saved model metadata."
        )

    threshold = float(metadata["classification_threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Saved classification threshold is invalid.")

    X_new = new_data[FEATURE_COLUMNS].copy()
    probabilities = model.predict_proba(X_new)[:, 1]

    output = pd.DataFrame()
    if ID_COLUMN in new_data.columns:
        output[ID_COLUMN] = new_data[ID_COLUMN]
    output["model_churn_score"] = np.round(probabilities, 6)
    output["predicted_churn_class_at_validation_threshold"] = (
        probabilities >= threshold
    ).astype(int)
    output["model_version"] = metadata["created_at_utc"]
    output["classification_threshold"] = threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Scored {len(output):,} new customer rows.")
    print(f"Output: {output_path}")
    print(
        "Caution: the supplied model is retrospective. This score is not a validated "
        "future-period churn probability until training and scoring use dated, "
        "point-in-time snapshots with a defined outcome window."
    )


if __name__ == "__main__":
    main()
