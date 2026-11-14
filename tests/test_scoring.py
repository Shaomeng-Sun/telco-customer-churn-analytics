import os
import subprocess
import sys

import pandas as pd

from telco_churn.config import (
    PROJECT_ROOT,
    METADATA_PATH,
)
from telco_churn.ml.common import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    load_modeling_data,
    load_json,
)


def test_scoring_script_runs(tmp_path):
    df = load_modeling_data()

    # Prepare five temporary records for a technical test.
    sample = df.sample(n=5, random_state=42)

    input_data = sample[[ID_COLUMN] + FEATURE_COLUMNS].copy()

    input_path = tmp_path / "scoring_input.csv"
    output_path = tmp_path / "scoring_output.csv"

    input_data.to_csv(input_path, index=False)

    # Configure Python to find the project package.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    # Run the existing scoring script.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "telco_churn.ml.score_new_customers",
            "--input",
            str(input_path),
            "--data",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.is_file()

    # Validate the generated scoring output.
    output = pd.read_csv(output_path)

    assert len(output) == 5
    assert ID_COLUMN in output.columns
    assert "model_churn_score" in output.columns
    assert "predicted_churn_class_at_validation_threshold" in output.columns
    assert "classification_threshold" in output.columns

    assert output["model_churn_score"].between(0, 1).all()
    assert output[
        "predicted_churn_class_at_validation_threshold"
    ].isin([0, 1]).all()

    metadata = load_json(METADATA_PATH)

    assert (
        output["classification_threshold"].iloc[0]
        == metadata["classification_threshold"]
    )

    assert output[ID_COLUMN].tolist() == input_data[ID_COLUMN].tolist()
