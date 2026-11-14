# Raw Data Quality & Profiling Report

- **Source:** `data/telco_warehouse.db | table: raw_telco_churn`
- **Generated:** 2026-11-14 01:23:21 UTC
- **Scope:** Raw SQLite table profiling only. The database is opened read-only and is not modified.

## 1. Dataset overview

| Metric | Result |
| --- | --- |
| Rows | 7,043 |
| Columns | 21 |
| Total cells | 147,903 |
| Exact duplicate rows | 0 |
| Conventional null cells | 0 |
| Whitespace-only cells | 11 |

### Expected columns

Expected columns missing from the file: None.

## 2. Missing-value assessment

Conventional nulls and whitespace-only strings are reported separately. A whitespace-only string may not be detected by a basic `DataFrame.isna()` check.

| Column | Null count | Whitespace-only count | Null % |
| --- | --- | --- | --- |
| TotalCharges | 0 | 11 | 0.000% |
| customerID | 0 | 0 | 0.000% |
| gender | 0 | 0 | 0.000% |
| SeniorCitizen | 0 | 0 | 0.000% |
| Partner | 0 | 0 | 0.000% |
| Dependents | 0 | 0 | 0.000% |
| tenure | 0 | 0 | 0.000% |
| PhoneService | 0 | 0 | 0.000% |
| MultipleLines | 0 | 0 | 0.000% |
| InternetService | 0 | 0 | 0.000% |
| OnlineSecurity | 0 | 0 | 0.000% |
| OnlineBackup | 0 | 0 | 0.000% |
| DeviceProtection | 0 | 0 | 0.000% |
| TechSupport | 0 | 0 | 0.000% |
| StreamingTV | 0 | 0 | 0.000% |
| StreamingMovies | 0 | 0 | 0.000% |
| Contract | 0 | 0 | 0.000% |
| PaperlessBilling | 0 | 0 | 0.000% |
| PaymentMethod | 0 | 0 | 0.000% |
| MonthlyCharges | 0 | 0 | 0.000% |
| Churn | 0 | 0 | 0.000% |

## 3. Duplicate assessment

| Check | Result |
| --- | --- |
| Exact duplicate rows | 0 |
| Customer ID column present | True |
| Blank customer IDs | 0 |
| Rows with a repeated customer ID | 0 |
| Distinct customer IDs repeated | 0 |

Repeated customer ID examples: None found.

## 4. Numeric-column assessment

| Column | Source dtype | Valid numeric | Blank/null | Invalid nonblank | Min | Median | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SeniorCitizen | int64 | 7043 | 0 | 0 | 0 | 0.0 | 1 |
| tenure | int64 | 7043 | 0 | 0 | 0 | 29.0 | 72 |
| MonthlyCharges | float64 | 7043 | 0 | 0 | 18.25 | 70.35 | 118.75 |
| TotalCharges | str | 7032 | 11 | 0 | 18.8 | 1397.475 | 8684.8 |

### Numeric validation issues

No numeric conversion or range issues were identified in the checked columns.

## 5. Categorical-value assessment

| Column | Distinct nonblank values | Observed values | Unexpected count |
| --- | --- | --- | --- |
| gender | 2 | female, male | 0 |
| Partner | 2 | no, yes | 0 |
| Dependents | 2 | no, yes | 0 |
| PhoneService | 2 | no, yes | 0 |
| MultipleLines | 3 | no, no phone service, yes | 0 |
| InternetService | 3 | dsl, fiber optic, no | 0 |
| OnlineSecurity | 3 | no, no internet service, yes | 0 |
| OnlineBackup | 3 | no, no internet service, yes | 0 |
| DeviceProtection | 3 | no, no internet service, yes | 0 |
| TechSupport | 3 | no, no internet service, yes | 0 |
| StreamingTV | 3 | no, no internet service, yes | 0 |
| StreamingMovies | 3 | no, no internet service, yes | 0 |
| Contract | 3 | month-to-month, one year, two year | 0 |
| PaperlessBilling | 2 | no, yes | 0 |
| PaymentMethod | 4 | bank transfer (automatic), credit card (automatic), electronic check, mailed check | 0 |
| Churn | 2 | no, yes | 0 |

### Unexpected categorical values

No unexpected values were identified in the checked categorical columns.

## 6. Dataset-specific cross-field checks

| Check | Count | Interpretation |
| --- | --- | --- |
| Blank TotalCharges with zero tenure | 11 | Review these records; in the supplied Telco dataset they are associated with newly recorded customers. |
| Blank TotalCharges with non-zero tenure | 0 | Investigate because blank accumulated charges are less readily explained by zero tenure. |
| Blank TotalCharges and Churn = No | 11 | Descriptive cross-check; do not automatically replace the blank with zero. |
| Blank TotalCharges and Churn = Yes | 0 | Investigate these records if any are present. |

## 7. Data-type inventory

| Column | Detected dtype | Non-null count |
| --- | --- | --- |
| customerID | str | 7043 |
| gender | str | 7043 |
| SeniorCitizen | int64 | 7043 |
| Partner | str | 7043 |
| Dependents | str | 7043 |
| tenure | int64 | 7043 |
| PhoneService | str | 7043 |
| MultipleLines | str | 7043 |
| InternetService | str | 7043 |
| OnlineSecurity | str | 7043 |
| OnlineBackup | str | 7043 |
| DeviceProtection | str | 7043 |
| TechSupport | str | 7043 |
| StreamingTV | str | 7043 |
| StreamingMovies | str | 7043 |
| Contract | str | 7043 |
| PaperlessBilling | str | 7043 |
| PaymentMethod | str | 7043 |
| MonthlyCharges | float64 | 7043 |
| TotalCharges | str | 7043 |
| Churn | str | 7043 |

## 8. Recommended handling

1. Keep the raw source file unchanged.
2. Trim leading/trailing whitespace in the dbt staging model.
3. Convert whitespace-only `TotalCharges` values to SQL `NULL`, not automatically to `0.0`.
4. Cast numeric fields explicitly and test values that cannot be converted.
5. Add dbt tests for unique customer IDs, required fields, accepted categorical values, non-negative tenure/charges, and fact-to-dimension relationships.
6. Document any retained missing values and handle them explicitly in the machine-learning preprocessing pipeline.

## 9. Scope and limitations

This report profiles the raw input file. It does not prove that all values are semantically correct, and it does not apply cleaning transformations. Business-specific assumptions should be reviewed before changing or imputing source values.
