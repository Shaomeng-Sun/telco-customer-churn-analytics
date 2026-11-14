import joblib

from telco_churn.config import MODEL_PATH, METADATA_PATH
from telco_churn.ml.common import (
    FEATURE_COLUMNS,
    load_json,
    load_modeling_data,
)


def test_saved_model_and_metadata_exist():
    assert MODEL_PATH.is_file()
    assert METADATA_PATH.is_file()


def test_saved_model_can_generate_probabilities():
    model = joblib.load(MODEL_PATH)
    metadata = load_json(METADATA_PATH)

    assert metadata["model_name"]
    assert metadata["feature_columns"] == FEATURE_COLUMNS

    threshold = float(metadata["classification_threshold"])
    assert 0 <= threshold <= 1

    df = load_modeling_data()

    X = df[FEATURE_COLUMNS].head(5)

    probabilities = model.predict_proba(X)[:, 1]

    assert len(probabilities) == 5
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
