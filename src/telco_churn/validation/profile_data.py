from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from telco_churn.config import (
    DB_PATH,
    DATA_QUALITY_REPORT_PATH,
    PROJECT_ROOT,
    resolve_project_path,
)

# Expected raw columns in the IBM/Kaggle Telco Customer Churn dataset.
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

CATEGORICAL_VALUES: dict[str, set[str]] = {
    "gender": {"female", "male"},
    "Partner": {"yes", "no"},
    "Dependents": {"yes", "no"},
    "PhoneService": {"yes", "no"},
    "MultipleLines": {"yes", "no", "no phone service"},
    "InternetService": {"dsl", "fiber optic", "no"},
    "OnlineSecurity": {"yes", "no", "no internet service"},
    "OnlineBackup": {"yes", "no", "no internet service"},
    "DeviceProtection": {"yes", "no", "no internet service"},
    "TechSupport": {"yes", "no", "no internet service"},
    "StreamingTV": {"yes", "no", "no internet service"},
    "StreamingMovies": {"yes", "no", "no internet service"},
    "Contract": {"month-to-month", "one year", "two year"},
    "PaperlessBilling": {"yes", "no"},
    "PaymentMethod": {
        "electronic check",
        "mailed check",
        "bank transfer (automatic)",
        "credit card (automatic)",
    },
    "Churn": {"yes", "no"},
}

NUMERIC_RULES: dict[str, dict[str, Any]] = {
    "SeniorCitizen": {
        "minimum": 0,
        "maximum": 1,
        "description": "Expected to be 0 or 1",
    },
    "tenure": {
        "minimum": 0,
        "maximum": None,
        "description": "Expected to be non-negative",
    },
    "MonthlyCharges": {
        "minimum": 0,
        "maximum": None,
        "description": "Expected to be non-negative",
    },
    "TotalCharges": {
        "minimum": 0,
        "maximum": None,
        "description": "Expected to be non-negative when present",
    },
}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Build a simple Markdown table, safely escaping pipe characters."""

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header_line = "| " + " | ".join(cell(value) for value in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in rows
    ]

    return "\n".join([header_line, separator, *body])


def read_raw_sqlite(
    database_path: Path,
    table_name: str,
) -> pd.DataFrame:
    """Read a raw SQLite table without modifying the database."""

    if not database_path.is_file():
        raise FileNotFoundError(
            f"SQLite database was not found: {database_path}. "
            "Run your ingestion script first."
        )

    connection_uri = (
        f"{database_path.resolve().as_uri()}?mode=ro"
    )

    try:
        connection = sqlite3.connect(
            connection_uri,
            uri=True,
        )

        try:
            table_exists = connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (table_name,),
            ).fetchone()

            if table_exists is None:
                raise ValueError(
                    f"Table '{table_name}' was not found "
                    f"in {database_path}"
                )

            quoted_table = '"' + table_name.replace('"', '""') + '"'

            return pd.read_sql_query(
                f"SELECT * FROM {quoted_table}",
                connection,
            )

        finally:
            connection.close()

    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Could not read SQLite database: {exc}"
        ) from exc


def profile_missing_values(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Count true nulls and whitespace-only strings separately."""

    rows: list[dict[str, Any]] = []
    whitespace_total = 0

    for column in df.columns:
        null_count = int(df[column].isna().sum())

        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            values = df[column].astype("string")
            whitespace_count = int(
                (
                    values.notna()
                    & values.str.strip().eq("")
                ).sum()
            )
        else:
            whitespace_count = 0

        whitespace_total += whitespace_count

        rows.append(
            {
                "column": column,
                "null_count": null_count,
                "whitespace_only_count": whitespace_count,
                "null_pct": (
                    round(null_count / len(df) * 100, 3)
                    if len(df)
                    else 0
                ),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        by=["null_count", "whitespace_only_count"],
        ascending=False,
    ).reset_index(drop=True)

    totals = {
        "null_cells": int(df.isna().sum().sum()),
        "whitespace_only_cells": whitespace_total,
    }

    return result, totals


def profile_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Check fully duplicated rows and duplicate customer identifiers."""

    full_duplicate_rows = int(df.duplicated().sum())

    result: dict[str, Any] = {
        "full_duplicate_rows": full_duplicate_rows,
        "customer_id_column_exists": "customerID" in df.columns,
        "blank_customer_ids": None,
        "duplicate_customer_id_rows": None,
        "number_of_repeated_customer_ids": None,
        "repeated_customer_id_examples": [],
    }

    if "customerID" in df.columns:
        ids = df["customerID"].astype("string").str.strip()

        blank_ids = ids.isna() | ids.eq("")

        duplicated_ids = (
            ids.notna()
            & ids.ne("")
            & ids.duplicated(keep=False)
        )

        repeated_values = ids[duplicated_ids].dropna().value_counts()

        result.update(
            {
                "blank_customer_ids": int(blank_ids.sum()),
                "duplicate_customer_id_rows": int(
                    ids[duplicated_ids].shape[0]
                ),
                "number_of_repeated_customer_ids": int(
                    repeated_values.shape[0]
                ),
                "repeated_customer_id_examples": [
                    f"{customer_id} ({count} rows)"
                    for customer_id, count in repeated_values.head(5).items()
                ],
            }
        )

    return result


def profile_numeric_columns(
    df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check numeric convertibility and simple range rules."""

    numeric_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for column, rule in NUMERIC_RULES.items():

        if column not in df.columns:
            issue_rows.append(
                {
                    "column": column,
                    "issue": "Required numeric column is missing",
                    "count": "-",
                    "examples": "-",
                }
            )
            continue

        raw = df[column].astype("string").str.strip()

        blank_mask = raw.isna() | raw.eq("")

        parsed = pd.to_numeric(
            raw.mask(blank_mask),
            errors="coerce",
        )

        invalid_mask = (~blank_mask) & parsed.isna()

        minimum = rule["minimum"]
        maximum = rule["maximum"]

        below_minimum = (
            parsed.notna() & (parsed < minimum)
            if minimum is not None
            else pd.Series(False, index=df.index)
        )

        above_maximum = (
            parsed.notna() & (parsed > maximum)
            if maximum is not None
            else pd.Series(False, index=df.index)
        )

        numeric_rows.append(
            {
                "column": column,
                "source_dtype": str(df[column].dtype),
                "valid_numeric_count": int(parsed.notna().sum()),
                "blank_or_null_count": int(blank_mask.sum()),
                "invalid_nonblank_count": int(invalid_mask.sum()),
                "minimum": (
                    parsed.min() if parsed.notna().any() else "-"
                ),
                "median": (
                    parsed.median() if parsed.notna().any() else "-"
                ),
                "maximum": (
                    parsed.max() if parsed.notna().any() else "-"
                ),
            }
        )

        for issue_name, mask in [
            ("Non-numeric nonblank values", invalid_mask),
            (f"Values below {minimum}", below_minimum),
            (f"Values above {maximum}", above_maximum),
        ]:
            count = int(mask.sum())

            if count:
                examples = (
                    raw[mask]
                    .dropna()
                    .astype(str)
                    .unique()[:5]
                    .tolist()
                )

                issue_rows.append(
                    {
                        "column": column,
                        "issue": issue_name,
                        "count": count,
                        "examples": ", ".join(examples),
                    }
                )

    return numeric_rows, issue_rows


def profile_categories(
    df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check observed categorical values against expected dataset values."""

    summary_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for column, allowed_values in CATEGORICAL_VALUES.items():

        if column not in df.columns:
            issue_rows.append(
                {
                    "column": column,
                    "issue": "Required categorical column is missing",
                    "count": "-",
                    "examples": "-",
                }
            )
            continue

        values = df[column].astype("string").str.strip()

        nonblank = values.notna() & values.ne("")

        normalized = values.str.casefold()

        observed = sorted(
            normalized[nonblank].dropna().unique().tolist()
        )

        unexpected_mask = nonblank & ~normalized.isin(allowed_values)

        unexpected_count = int(unexpected_mask.sum())

        summary_rows.append(
            {
                "column": column,
                "distinct_nonblank_values": len(observed),
                "observed_values": ", ".join(observed),
                "unexpected_value_count": unexpected_count,
            }
        )

        if unexpected_count:
            examples = (
                values[unexpected_mask]
                .dropna()
                .unique()[:5]
                .tolist()
            )

            issue_rows.append(
                {
                    "column": column,
                    "issue": "Unexpected categorical values",
                    "count": unexpected_count,
                    "examples": ", ".join(map(str, examples)),
                }
            )

    return summary_rows, issue_rows


def profile_required_columns(df: pd.DataFrame) -> list[str]:
    """Return expected columns that are absent from the raw file."""

    return [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]


def profile_telco_logic(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check a small set of dataset-specific cross-column conditions."""

    findings: list[dict[str, Any]] = []

    required = {"TotalCharges", "tenure", "Churn"}

    if not required.issubset(df.columns):
        return findings

    total_raw = df["TotalCharges"].astype("string").str.strip()

    total_blank = total_raw.isna() | total_raw.eq("")

    tenure = pd.to_numeric(
        df["tenure"],
        errors="coerce",
    )

    churn = (
        df["Churn"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    blank_total_zero_tenure = total_blank & tenure.eq(0)

    blank_total_nonzero_tenure = (
        total_blank & tenure.ne(0) & tenure.notna()
    )

    blank_total_active = total_blank & churn.eq("no")

    blank_total_churned = total_blank & churn.eq("yes")

    findings.extend(
        [
            {
                "check": "Blank TotalCharges with zero tenure",
                "count": int(blank_total_zero_tenure.sum()),
                "interpretation": (
                    "Review these records; in the supplied Telco dataset "
                    "they are associated with newly recorded customers."
                ),
            },
            {
                "check": "Blank TotalCharges with non-zero tenure",
                "count": int(blank_total_nonzero_tenure.sum()),
                "interpretation": (
                    "Investigate because blank accumulated charges are "
                    "less readily explained by zero tenure."
                ),
            },
            {
                "check": "Blank TotalCharges and Churn = No",
                "count": int(blank_total_active.sum()),
                "interpretation": (
                    "Descriptive cross-check; do not automatically "
                    "replace the blank with zero."
                ),
            },
            {
                "check": "Blank TotalCharges and Churn = Yes",
                "count": int(blank_total_churned.sum()),
                "interpretation": (
                    "Investigate these records if any are present."
                ),
            },
        ]
    )

    return findings


def portable_path(path: Path) -> str:
    """Return a project-relative path for reports, without exposing local paths."""

    try:
        return (
            path.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix()
        )

    except ValueError:
        # For files outside the project, show only the filename.
        return path.name


def build_report(
    df: pd.DataFrame,
    source_description: str,
) -> str:
    """Create a Markdown data profiling report."""

    missing_df, missing_totals = profile_missing_values(df)

    duplicates = profile_duplicates(df)

    numeric_rows, numeric_issues = profile_numeric_columns(df)

    category_rows, category_issues = profile_categories(df)

    missing_columns = profile_required_columns(df)

    logic_rows = profile_telco_logic(df)

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    total_cells = int(df.shape[0] * df.shape[1])

    missing_column_rows = [
        [
            r["column"],
            r["null_count"],
            r["whitespace_only_count"],
            f"{r['null_pct']:.3f}%",
        ]
        for r in missing_df.to_dict(orient="records")
    ]

    numeric_table_rows = [
        [
            r["column"],
            r["source_dtype"],
            r["valid_numeric_count"],
            r["blank_or_null_count"],
            r["invalid_nonblank_count"],
            r["minimum"],
            r["median"],
            r["maximum"],
        ]
        for r in numeric_rows
    ]

    category_table_rows = [
        [
            r["column"],
            r["distinct_nonblank_values"],
            r["observed_values"],
            r["unexpected_value_count"],
        ]
        for r in category_rows
    ]

    numeric_issue_table = [
        [
            r["column"],
            r["issue"],
            r["count"],
            r["examples"],
        ]
        for r in numeric_issues
    ]

    category_issue_table = [
        [
            r["column"],
            r["issue"],
            r["count"],
            r["examples"],
        ]
        for r in category_issues
    ]

    logic_table = [
        [r["check"], r["count"], r["interpretation"]]
        for r in logic_rows
    ]

    report_parts = [
        "# Raw Data Quality & Profiling Report",
        "",
        f"- **Source:** `{source_description}`",
        f"- **Generated:** {generated_at}",
        (
            "- **Scope:** Raw SQLite table profiling only. "
            "The database is opened read-only and is not modified."
        ),
        "",
        "## 1. Dataset overview",
        "",
        markdown_table(
            ["Metric", "Result"],
            [
                ["Rows", f"{df.shape[0]:,}"],
                ["Columns", f"{df.shape[1]:,}"],
                ["Total cells", f"{total_cells:,}"],
                [
                    "Exact duplicate rows",
                    f"{duplicates['full_duplicate_rows']:,}",
                ],
                [
                    "Conventional null cells",
                    f"{missing_totals['null_cells']:,}",
                ],
                [
                    "Whitespace-only cells",
                    f"{missing_totals['whitespace_only_cells']:,}",
                ],
            ],
        ),
        "",
        "### Expected columns",
        "",
        (
            "Expected columns missing from the file: "
            f"{', '.join(missing_columns) if missing_columns else 'None'}."
        ),
        "",
        "## 2. Missing-value assessment",
        "",
        (
            "Conventional nulls and whitespace-only strings are reported "
            "separately. A whitespace-only string may not be detected "
            "by a basic `DataFrame.isna()` check."
        ),
        "",
        markdown_table(
            [
                "Column",
                "Null count",
                "Whitespace-only count",
                "Null %",
            ],
            missing_column_rows,
        ),
        "",
        "## 3. Duplicate assessment",
        "",
        markdown_table(
            ["Check", "Result"],
            [
                [
                    "Exact duplicate rows",
                    duplicates["full_duplicate_rows"],
                ],
                [
                    "Customer ID column present",
                    duplicates["customer_id_column_exists"],
                ],
                [
                    "Blank customer IDs",
                    duplicates["blank_customer_ids"],
                ],
                [
                    "Rows with a repeated customer ID",
                    duplicates["duplicate_customer_id_rows"],
                ],
                [
                    "Distinct customer IDs repeated",
                    duplicates["number_of_repeated_customer_ids"],
                ],
            ],
        ),
        "",
        (
            "Repeated customer ID examples: "
            + (
                ", ".join(duplicates["repeated_customer_id_examples"])
                or "None found."
            )
        ),
        "",
        "## 4. Numeric-column assessment",
        "",
        (
            markdown_table(
                [
                    "Column",
                    "Source dtype",
                    "Valid numeric",
                    "Blank/null",
                    "Invalid nonblank",
                    "Min",
                    "Median",
                    "Max",
                ],
                numeric_table_rows,
            )
            if numeric_table_rows
            else "No numeric columns were assessed."
        ),
        "",
        "### Numeric validation issues",
        "",
        (
            markdown_table(
                ["Column", "Issue", "Count", "Examples"],
                numeric_issue_table,
            )
            if numeric_issue_table
            else (
                "No numeric conversion or range issues were identified "
                "in the checked columns."
            )
        ),
        "",
        "## 5. Categorical-value assessment",
        "",
        (
            markdown_table(
                [
                    "Column",
                    "Distinct nonblank values",
                    "Observed values",
                    "Unexpected count",
                ],
                category_table_rows,
            )
            if category_table_rows
            else "No categorical columns were assessed."
        ),
        "",
        "### Unexpected categorical values",
        "",
        (
            markdown_table(
                ["Column", "Issue", "Count", "Examples"],
                category_issue_table,
            )
            if category_issue_table
            else "No unexpected values were identified in the checked categorical columns."
        ),
        "",
        "## 6. Dataset-specific cross-field checks",
        "",
        (
            markdown_table(
                ["Check", "Count", "Interpretation"],
                logic_table,
            )
            if logic_table
            else (
                "Required columns for the Telco-specific checks "
                "were not all present."
            )
        ),
        "",
        "## 7. Data-type inventory",
        "",
        markdown_table(
            [
                "Column",
                "Detected dtype",
                "Non-null count",
            ],
            [
                [
                    column,
                    str(df[column].dtype),
                    int(df[column].notna().sum()),
                ]
                for column in df.columns
            ],
        ),
        "",
        "## 8. Recommended handling",
        "",
        "1. Keep the raw source file unchanged.",
        "2. Trim leading/trailing whitespace in the dbt staging model.",
        (
            "3. Convert whitespace-only `TotalCharges` values to SQL "
            "`NULL`, not automatically to `0.0`."
        ),
        (
            "4. Cast numeric fields explicitly and test values "
            "that cannot be converted."
        ),
        (
            "5. Add dbt tests for unique customer IDs, required fields, "
            "accepted categorical values, non-negative tenure/charges, "
            "and fact-to-dimension relationships."
        ),
        (
            "6. Document any retained missing values and handle them "
            "explicitly in the machine-learning preprocessing pipeline."
        ),
        "",
        "## 9. Scope and limitations",
        "",
        (
            "This report profiles the raw input file. It does not prove "
            "that all values are semantically correct, and it does not "
            "apply cleaning transformations. Business-specific assumptions "
            "should be reviewed before changing or imputing source values."
        ),
        "",
    ]

    return "\n".join(report_parts)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile the raw Telco Customer Churn SQLite table "
            "and write a Markdown report."
        )
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DB_PATH,
        help="SQLite database path.",
    )

    parser.add_argument(
        "--table",
        default="raw_telco_churn",
        help="Raw SQLite table to profile.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_QUALITY_REPORT_PATH,
        help="Markdown report output path.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the profiling workflow and save the report."""

    args = parse_args()

    # Resolve database and report paths using central configuration.
    database_path = resolve_project_path(args.database)
    output_path = resolve_project_path(args.output)

    try:
        print(f"Reading SQLite database: {portable_path(database_path)}")
        print(f"Reading table: {args.table}")

        df = read_raw_sqlite(database_path, args.table)

        source_description = (
            f"{portable_path(database_path)} | table: {args.table}"
        )

        report = build_report(df, source_description)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            report,
            encoding="utf-8",
        )

        print("\nProfiling completed.")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns):,}")
        print(f"Report saved to: {portable_path(output_path)}")
        print(
            "The source SQLite database was opened read-only "
            "and was not modified."
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        pd.errors.EmptyDataError,
        OSError,
    ) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()