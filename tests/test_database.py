import sqlite3

from telco_churn.config import DB_PATH


REQUIRED_TABLES = {
    "fact_customer_churn",
    "dim_customer",
    "dim_contract_billing",
    "dim_services",
}


def test_database_exists():
    assert DB_PATH.is_file(), f"Database not found: {DB_PATH}"


def test_required_dbt_tables_exist():
    with sqlite3.connect(
        f"file:{DB_PATH.as_posix()}?mode=ro",
        uri=True,
    ) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert REQUIRED_TABLES.issubset(tables)


def test_fact_table_has_expected_customers():
    with sqlite3.connect(
        f"file:{DB_PATH.as_posix()}?mode=ro",
        uri=True,
    ) as conn:
        row_count = conn.execute(
            "SELECT COUNT(*) FROM fact_customer_churn"
        ).fetchone()[0]

        unique_count = conn.execute(
            "SELECT COUNT(DISTINCT customer_id) "
            "FROM fact_customer_churn"
        ).fetchone()[0]

    raw_count = conn.execute(
        "SELECT COUNT(*) FROM raw_telco_churn"
    ).fetchone()[0]

    assert row_count == raw_count
    assert unique_count == raw_count
