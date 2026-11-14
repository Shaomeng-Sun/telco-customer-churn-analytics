"""Evaluate the selected model once on the held-out test partition."""

from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from telco_churn.config import (
    PROJECT_ROOT,
    ML_FIGURES_DIR,
    ML_OUTPUT_DIR,
    METADATA_PATH,
    MODEL_PATH,
)
from telco_churn.ml.common import load_json, load_modeling_data, make_train_validation_test_split


def main() -> None:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("Run train.py first to create the model and metadata.")

    
    # Load the saved model and training metadata.
    model = joblib.load(MODEL_PATH)
    metadata = load_json(METADATA_PATH)

    # Retrieve the random seed used during model training.
    if "random_state" not in metadata:
        raise KeyError(
            "random_state is missing from model metadata. "
            "Please run train.py again."
        )

    random_state = int(metadata["random_state"])

    # Recreate the original training/validation/test split.
    df = load_modeling_data()

    splits = make_train_validation_test_split(
        df,
        random_state=random_state,
    )

    X_test, y_test = splits["X_test"], splits["y_test"]

    # Retrieve the classification threshold selected during training.
    threshold = float(metadata["classification_threshold"])


    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "model_name": metadata["model_name"],
        "threshold": threshold,
        "test_rows": int(len(y_test)),
        "test_churn_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision_pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }

    ML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ML_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (ML_OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_test, predictions, target_names=["Not churned", "Churned"], zero_division=0
    )
    report_text = [
        "# Churn Model Evaluation",
        "",
        "Evaluation on the held-out test partition. The threshold was selected using validation data only.",
        "",
        f"- Model: **{metadata['model_name']}**",
        f"- Threshold: **{threshold:.2f}**",
        f"- Test records: **{len(y_test):,}**",
        "",
        "## Metrics",
        "",
        "| Metric | Test result |",
        "|---|---:|",
    ]
    for key in [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "roc_auc", "average_precision_pr_auc", "brier_score"
    ]:
        report_text.append(f"| {key.replace('_', ' ').title()} | {metrics[key]:.4f} |")
    report_text.extend(["", "## Classification report", "", "```text", report, "```", ""])
    report_text.extend([
        "> **Scope limitation:** These metrics describe retrospective classification of the supplied customer snapshot. "
        "They do not establish performance for a future 30-day churn prediction task."
    ])
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "model_evaluation.md").write_text(
        "\n".join(report_text),
        encoding="utf-8",
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["Not churned", "Churned"],
        values_format="d",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {metadata['model_name']}")
    fig.tight_layout()
    fig.savefig(ML_FIGURES_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ML_FIGURES_DIR / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Average precision = {metrics['average_precision_pr_auc']:.3f}")
    ax.axhline(y_test.mean(), linestyle="--", label=f"Churn prevalence = {y_test.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ML_FIGURES_DIR / "precision_recall_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Local test predictions without customer identifiers.
    pd.DataFrame(
        {
            "actual_churn": y_test.to_numpy(),
            "predicted_churn_probability": probabilities,
            "predicted_churn_class": predictions,
        }
    ).to_csv(ML_OUTPUT_DIR / "test_predictions.csv", index=False)

    print("Held-out test evaluation complete.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\nText Report: {PROJECT_ROOT / 'docs' / 'model_evaluation.md'}")
    print(f"Figures Report: {ML_FIGURES_DIR}")


if __name__ == "__main__":
    main()
