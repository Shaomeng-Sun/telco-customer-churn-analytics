"""Compare churn classifiers, select a model, and choose a threshold."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer, precision_recall_fscore_support, precision_score, recall_score
import joblib

from telco_churn.config import ML_OUTPUT_DIR, METADATA_PATH, MODEL_DIR, MODEL_PATH
from telco_churn.ml.common import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    load_modeling_data,
    make_train_validation_test_split,
    save_json,
)

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def candidate_models() -> dict[str, object]:
    """Return a small, interpretable set of baseline and candidate models."""
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose an F1-maximizing threshold using validation data only."""
    rows = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            predictions,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "customers_flagged": int(predictions.sum()),
            }
        )

    table = pd.DataFrame(rows)
    # Tie-break equal F1 values in favour of higher recall.
    best = table.sort_values(
        ["f1", "recall", "precision"], ascending=[False, False, False]
    ).iloc[0]
    return float(best["threshold"]), table


def main() -> None:
    print("Loading customer data from the dbt Star Schema...")
    df = load_modeling_data()
    splits = make_train_validation_test_split(df, RANDOM_STATE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Churn rate: {df[TARGET_COLUMN].mean():.2%}")
    print(
        "Split sizes: "
        f"train={len(splits['X_train']):,}, "
        f"validation={len(splits['X_validation']):,}, "
        f"test={len(splits['X_test']):,}"
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows = []

    print("\nComparing models with stratified cross-validation...")
    for model_name, classifier in candidate_models().items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)]
        )
        scores = cross_validate(
            pipeline,
            splits["X_train"],
            splits["y_train"],
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        row = {"model": model_name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        comparison_rows.append(row)
        print(
            f"{model_name:24s} "
            f"CV PR-AUC={row['average_precision_mean']:.4f} "
            f"(+/- {row['average_precision_std']:.4f})"
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["average_precision_mean", "roc_auc_mean"], ascending=False
    )

    ML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        ML_OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )
    best_model_name = str(comparison.iloc[0]["model"])

    # Fit the selected model on the training partition only.
    selected_classifier = candidate_models()[best_model_name]
    selected_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", selected_classifier),
        ]
    )
    selected_pipeline.fit(splits["X_train"], splits["y_train"])

    # Use validation data to choose the operating threshold. The test set remains untouched.
    validation_probabilities = selected_pipeline.predict_proba(
        splits["X_validation"]
    )[:, 1]
    threshold, threshold_table = choose_threshold(
        splits["y_validation"], validation_probabilities
    )
    threshold_table.to_csv(ML_OUTPUT_DIR / "validation_threshold_analysis.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_PATH)

    metadata = {
        "model_name": best_model_name,
        "model_file": str(MODEL_PATH.relative_to(MODEL_DIR.parent.parent)),
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classification_threshold": threshold,
        "model_selection_metric": "mean 5-fold validation average precision (PR-AUC)",
        "cross_validation_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(splits["X_train"])),
        "validation_rows": int(len(splits["X_validation"])),
        "test_rows": int(len(splits["X_test"])),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": (
            "Retrospective classification of the recorded churn label in the supplied snapshot. "
            "Not validated as a prospective 30-day churn prediction model."
        ),
        "feature_limitations": (
            "tenure_months is included as a snapshot feature. total_charges is excluded "
            "from the primary model because it is cumulative and tied to observed duration."
        ),
    }
    save_json(metadata, METADATA_PATH)

    best_row = comparison.iloc[0]
    print("\nSelected model:", best_model_name)
    print(f"Mean CV PR-AUC: {best_row['average_precision_mean']:.4f}")
    print(f"Validation-selected threshold: {threshold:.2f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")
    print("\nImportant: this is retrospective classification, not validated future churn forecasting.")


if __name__ == "__main__":
    main()
