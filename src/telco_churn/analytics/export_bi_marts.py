from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from telco_churn.config import (
    PROJECT_ROOT,
    DB_PATH,
    BI_OUTPUT_DIR,
    CONTRACT_PAYMENT_SQL,
    TENURE_ANALYSIS_SQL,
    SERVICE_BUNDLE_SQL,
    MASTER_VIEW_SQL,
    resolve_project_path,
)


ANALYSIS_QUERIES = {
    "1_contract_payment_risk": CONTRACT_PAYMENT_SQL,
    "2_tenure_cohort_curve": TENURE_ANALYSIS_SQL,
    "3_service_bundle_retention": SERVICE_BUNDLE_SQL,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional database and output-directory overrides."""
    default_db = str(DB_PATH)
    default_output = str(BI_OUTPUT_DIR)

    parser = argparse.ArgumentParser(
        description="Run Telco churn business analytics SQL and export BI datasets."
    )
    parser.add_argument(
        "--database",
        default=default_db,
        help=(
            "SQLite database path. Relative paths are resolved from the project "
            "root. Can also be set with TELCO_DB_PATH."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help=(
            "Directory for generated CSV files. Relative paths are resolved "
            "from the project root. Can also be set with TELCO_BI_OUTPUT_DIR."
        ),
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Maximum number of rows to display for each query (default: 10).",
    )
    return parser.parse_args(argv)


def create_read_only_connection(database_path: Path) -> sqlite3.Connection:
    """Open an existing SQLite database without permitting modifications."""
    if not database_path.is_file():
        raise FileNotFoundError(
            f"SQLite database not found: {database_path}\n"
            "Run the ingestion script first or provide --database."
        )

    connection_uri = f"{database_path.as_uri()}?mode=ro"
    connection = sqlite3.connect(connection_uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def read_sql_file(sql_path: Path) -> str:
    """Read a SQL file and reject empty query files."""
    if not sql_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql = sql_path.read_text(encoding="utf-8").strip()
    if not sql:
        raise ValueError(f"SQL file is empty: {sql_path}")
    return sql


def display_preview(name: str, dataframe: pd.DataFrame, preview_rows: int) -> None:
    """Print a compact and readable query summary."""
    print(f"\n{'=' * 78}")
    print(name.replace("_", " ").title())
    print(f"{'=' * 78}")
    print(f"Rows returned: {len(dataframe):,}")
    print(f"Columns returned: {len(dataframe.columns):,}")

    if dataframe.empty:
        print("Query returned no rows.")
        return

    print("\nPreview:")
    print(
        dataframe.head(preview_rows).to_string(
            index=False,
            max_cols=20,
            max_colwidth=32,
        )
    )


def validate_master_view(
    dataframe: pd.DataFrame,
    connection: sqlite3.Connection,
) -> None:
    """Check that the denormalized export preserves the fact-table grain."""
    required_columns = {
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "churn_flag",
        "contract_type",
        "payment_method",
        "internet_service",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Master view is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if dataframe["customer_id"].isna().any():
        raise ValueError("Master view contains NULL customer_id values.")

    duplicate_count = int(dataframe["customer_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Master view contains {duplicate_count:,} duplicate customer IDs. "
            "Check uniqueness of dimension keys and fact-table grain."
        )

    fact_count = int(
        pd.read_sql_query(
            "SELECT COUNT(*) AS row_count FROM fact_customer_churn",
            connection,
        ).iloc[0]["row_count"]
    )

    if len(dataframe) != fact_count:
        raise ValueError(
            "Master view row count does not match fact_customer_churn: "
            f"master view={len(dataframe):,}, fact table={fact_count:,}. "
            "Check dimension join cardinality."
        )

    invalid_churn_values = ~dataframe["churn_flag"].isin([0, 1])
    if invalid_churn_values.any():
        raise ValueError("Master view contains churn_flag values other than 0 or 1.")


def export_query(
    connection: sqlite3.Connection,
    sql_path: Path,
    output_path: Path,
    preview_rows: int,
) -> pd.DataFrame:
    """Execute one SQL file and export its result to CSV."""
    sql = read_sql_file(sql_path)
    dataframe = pd.read_sql_query(sql, connection)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")

    display_preview(sql_path.stem, dataframe, preview_rows)
    print(f"\nExported: {output_path}")
    return dataframe


def run_business_analytics(
    database_path: Path,
    output_dir: Path,
    preview_rows: int = 10,
) -> None:
    """Run all analysis queries and export the BI master view."""
    if preview_rows < 0:
        raise ValueError("--preview-rows must be zero or a positive integer.")

    output_dir.mkdir(parents=True, exist_ok=True)
    connection = create_read_only_connection(database_path)

    try:
        print(f"Database: {database_path}")
        print(f"Output directory: {output_dir}")

        # Export aggregate business-analysis datasets.
        for output_name, sql_path in ANALYSIS_QUERIES.items():
            export_query(
                connection=connection,
                sql_path=sql_path,
                output_path=output_dir / f"{output_name}.csv",
                preview_rows=preview_rows,
            )

        # Export the customer-level master dataset after validating its grain.
        master_dataframe = pd.read_sql_query(
            read_sql_file(MASTER_VIEW_SQL),
            connection,
        )
        validate_master_view(master_dataframe, connection)

        master_output = output_dir / "bi_master_view.csv"
        master_dataframe.to_csv(
            master_output,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"\n{'=' * 78}")
        print("BI MASTER VIEW")
        print(f"{'=' * 78}")
        print(f"Customer records exported: {len(master_dataframe):,}")
        print(f"Columns exported: {len(master_dataframe.columns):,}")
        print(f"Unique customer IDs: {master_dataframe['customer_id'].nunique():,}")
        print(f"Exported: {master_output}")

    finally:
        connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    database_path = resolve_project_path(args.database)
    output_dir = resolve_project_path(args.output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        run_business_analytics(
            database_path=database_path,
            output_dir=output_dir,
            preview_rows=args.preview_rows,
        )
    except (FileNotFoundError, ValueError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        logging.error("Business analytics export failed: %s", exc)
        return 1
    except OSError as exc:
        logging.error("File operation failed: %s", exc)
        return 1

    print("\nAll business analytics exports completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
