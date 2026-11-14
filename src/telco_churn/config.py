
"""Central configuration and path management."""

from __future__ import annotations

import os
from pathlib import Path


# --------------------------------------------------
# 1. PROJECT ROOT
# --------------------------------------------------

MARKER = ".project-root"

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "venv312",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "target",
    "dbt_packages",
    "node_modules",
}


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by searching parent directories for .project-root."""

    current = (start or Path(__file__)).expanduser().resolve()

    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        if (directory / MARKER).is_file():
            return directory

    raise FileNotFoundError(
        "Project root not found. "
        "Ensure .project-root exists in the main project folder."
    )


PROJECT_ROOT = find_project_root()


# --------------------------------------------------
# 2. GENERAL PATH RESOLUTION
# --------------------------------------------------

def resolve_project_path(value: str | Path) -> Path:
    """Resolve absolute paths or paths relative to PROJECT_ROOT."""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# --------------------------------------------------
# 3. AUTOMATIC FILE DISCOVERY
# --------------------------------------------------

def find_project_file(filename: str) -> Path | None:
    """
    Find a file anywhere inside the project by its filename.

    Returns None if no file is found.
    Raises an error if multiple matching files exist.
    """

    matches = []

    for current, directories, files in os.walk(PROJECT_ROOT):

        directories[:] = [
            directory
            for directory in directories
            if directory not in IGNORED_DIRECTORIES
        ]

        if filename in files:
            matches.append(Path(current) / filename)

    if len(matches) > 1:
        locations = "\n".join(str(path) for path in matches)

        raise RuntimeError(
            f"Multiple files named '{filename}' were found:\n"
            f"{locations}\n"
            "Keep one authoritative copy or configure an explicit path."
        )

    return matches[0] if matches else None


def locate_project_file(
    filename: str,
    fallback_relative: str,
) -> Path:
    """
    Locate a file automatically.
    Use the conventional relative location if it does not exist yet.
    """

    discovered = find_project_file(filename)

    if discovered is not None:
        return discovered

    return resolve_project_path(fallback_relative)


# --------------------------------------------------
# 4. DATABASE CONFIGURATION
# --------------------------------------------------

def locate_database() -> Path:
    """Find the existing warehouse or use its default creation location."""

    override = os.getenv("TELCO_DB_PATH")

    if override:
        return resolve_project_path(override)

    existing_database = find_project_file("telco_warehouse.db")

    if existing_database is not None:
        return existing_database.resolve()

    # Default location when ingestion has not yet created the database.
    return PROJECT_ROOT / "data" / "telco_warehouse.db"


DB_PATH = locate_database()


# --------------------------------------------------
# 5. SQL FILE LOCATIONS
# --------------------------------------------------

CONTRACT_PAYMENT_SQL = locate_project_file(
    "01_contract_payment_risk.sql",
    "analytics/01_contract_payment_risk.sql",
)

TENURE_ANALYSIS_SQL = locate_project_file(
    "02_tenure_cohort_curve.sql",
    "analytics/02_tenure_cohort_curve.sql",
)

SERVICE_BUNDLE_SQL = locate_project_file(
    "03_service_bundle_retention.sql",
    "analytics/03_service_bundle_retention.sql",
)

MASTER_VIEW_SQL = locate_project_file(
    "bi_master_view.sql",
    "analytics/bi_master_view.sql",
)

TRAINING_SQL_PATH = locate_project_file(
    "customer_training_dataset.sql",
    "analytics/ml/customer_training_dataset.sql",
)


# --------------------------------------------------
# 6. OUTPUT DIRECTORIES
# --------------------------------------------------

# BI exports are saved next to the SQLite database.
BI_OUTPUT_DIR = resolve_project_path(
    os.getenv(
        "TELCO_BI_OUTPUT_DIR",
        str(DB_PATH.parent / "bi_exports"),
    )
)

DATA_QUALITY_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "data_quality_report.md"
)

MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

ML_OUTPUT_DIR = PROJECT_ROOT / "data" / "ml"
ML_FIGURES_DIR = PROJECT_ROOT / "reports" / "ml"