"""Create global and local SHAP explanations for the selected classifier."""

from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from telco_churn.config import ML_FIGURES_DIR, METADATA_PATH, MODEL_PATH
from telco_churn.ml.common import load_json, load_modeling_data, make_train_validation_test_split


def select_positive_class_values(shap_values, classifier):
    """Return SHAP values for class 1 if the explainer returns class-wise values."""
    values = np.asarray(shap_values.values)
    base_values = np.asarray(shap_values.base_values)
    classes = list(classifier.classes_)
    positive_index = classes.index(1)

    if values.ndim == 3:
        values = values[:, :, positive_index]
    if base_values.ndim == 2:
        base_values = base_values[:, positive_index]
    elif base_values.ndim == 1 and len(base_values) == 2 and values.shape[0] != 2:
        # This branch is uncommon for current SHAP/sklearn combinations.
        base_values = np.repeat(base_values[positive_index], values.shape[0])

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=shap_values.data,
        feature_names=shap_values.feature_names,
    )


def main() -> None:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("Run train.py first to create the model and metadata.")

    
    # Load the saved model and training metadata.
    metadata = load_json(METADATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

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


    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_train_transformed = np.asarray(preprocessor.transform(splits["X_train"]))
    X_test_transformed = np.asarray(preprocessor.transform(splits["X_test"]))
    feature_names = list(preprocessor.get_feature_names_out())

    # Use a small sample to keep SHAP fast and reproducible.
    rng = np.random.default_rng(random_state)
    background_count = min(30, len(X_train_transformed))
    explain_count = min(40, len(X_test_transformed))
    background_idx = rng.choice(len(X_train_transformed), background_count, replace=False)
    explain_idx = rng.choice(len(X_test_transformed), explain_count, replace=False)
    background = X_train_transformed[background_idx]
    explain_data = X_test_transformed[explain_idx]

    print(f"Explaining model: {metadata['model_name']}")
    print(f"Background rows: {background_count}; explained rows: {explain_count}")

    # Prefer fast, model-specific explainers. Logistic Regression SHAP values
    # explain the model's log-odds output; tree models can explain class probability.
    if isinstance(classifier, LogisticRegression):
        # Exact additive SHAP decomposition for a linear model under the
        # independent-feature assumption. Values are on the log-odds scale.
        # One-hot features are correlated by construction, so interpret this
        # as a practical global explanation rather than a causal attribution.
        coef = classifier.coef_[0]
        background_mean = background.mean(axis=0)
        base_value = float(classifier.intercept_[0] + background_mean @ coef)
        linear_values = (explain_data - background_mean) * coef
        raw_explanation = shap.Explanation(
            values=linear_values,
            base_values=np.full(len(explain_data), base_value),
            data=explain_data,
            feature_names=feature_names,
        )
        output_description = "model log-odds (positive churn class)"
    elif isinstance(classifier, RandomForestClassifier):
        explainer = shap.TreeExplainer(
            classifier,
            data=background,
            model_output="probability",
            feature_perturbation="interventional",
        )
        raw_explanation = explainer(explain_data)
        output_description = "predicted churn probability"
    else:
        # Permutation SHAP is a general fallback for estimators without a
        # supported model-specific explainer. The sample is intentionally small.
        positive_class_index = list(classifier.classes_).index(1)

        def predict_churn_probability(values):
            return classifier.predict_proba(np.asarray(values))[:, positive_class_index]

        masker = shap.maskers.Independent(background, max_samples=background_count)
        explainer = shap.Explainer(
            predict_churn_probability,
            masker,
            algorithm="permutation",
            feature_names=feature_names,
        )
        max_evals = 2 * len(feature_names) + 1
        raw_explanation = explainer(explain_data, max_evals=max_evals)
        output_description = "predicted churn probability"

    explanation = select_positive_class_values(raw_explanation, classifier)
    ML_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title(f"SHAP Summary — Contribution to {output_description}")
    plt.tight_layout()
    plt.savefig(ML_FIGURES_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("Mean Absolute SHAP Importance")
    plt.tight_layout()
    plt.savefig(ML_FIGURES_DIR / "shap_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Select the highest-scored example among the explained test rows.
    positive_class_index = list(classifier.classes_).index(1)
    sample_probabilities = classifier.predict_proba(explain_data)[:, positive_class_index]
    selected_idx = int(np.argmax(sample_probabilities))
    shap.plots.waterfall(explanation[selected_idx], max_display=15, show=False)
    plt.title("Local SHAP Explanation — Highest-Scored Test Example")
    plt.tight_layout()
    plt.savefig(ML_FIGURES_DIR / "shap_waterfall_example.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"SHAP explanations saved to: {ML_FIGURES_DIR}")
    print(f"Explained output: {output_description}")
    print("SHAP values explain model output contributions; they do not establish causality.")


if __name__ == "__main__":
    main()
