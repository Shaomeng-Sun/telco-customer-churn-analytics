# Telco Customer Churn Project — Methodology

## 1. Project overview

This project develops an end-to-end analytical workflow for exploring customer churn in a telecommunications customer dataset. It combines data ingestion, data-quality checks, SQL transformations with dbt, business analysis, machine-learning classification, model explainability, and a reusable scoring interface.

The project has two main objectives:

1. **Business analytics:** describe customer churn patterns and identify customer segments that show different observed churn rates.
2. **Machine learning:** train and evaluate a model that classifies the recorded churn outcome in the supplied customer snapshot, and prepare the saved model for reuse when a compatible customer dataset becomes available.

### Scope and interpretation

The source data is a customer snapshot with a recorded churn label. It does not contain the dated customer snapshots and subsequent outcome windows needed to establish whether a customer will churn within a defined future period (for example, the next 30 days).

Accordingly, the current machine-learning task is **retrospective churn classification**. It should not be described as a validated prospective churn forecast or a 30-day churn prediction model.

## 2. Dataset

The project uses the Telco Customer Churn dataset, containing **7,043 customer records and 21 original columns**.

The target variable is `Churn`, represented in the analytical and modelling data as the binary field `churn_flag`:

- `1`: the recorded customer outcome is churn.
- `0`: the recorded customer outcome is not churn.

The dataset contains **1,869 churned customers**, corresponding to an overall observed churn rate of approximately **26.54%**.

Each customer is identified by `customerID` in the source data and `customer_id` in the project data model. The customer-level analysis and modelling datasets are expected to retain one row per customer.

### Data-quality observations

The initial data inspection found:

- 7,043 rows and 21 columns.
- Unique customer identifiers.
- No exact duplicate rows.
- 11 `TotalCharges` values containing whitespace rather than a numeric value. These records have zero tenure and a recorded non-churn outcome.

Whitespace-only charge values should be treated as missing values during transformation, rather than interpreted as zero. The raw source should remain available so that transformation decisions are traceable.

## 3. Project architecture and data workflow

The project is organized into separate components so that ingestion, validation, transformation, analytics, and machine learning can be run and maintained independently.

### 3.1 Data ingestion

The ingestion module downloads or reads the source data and loads it into the project's SQLite database. The database location is managed centrally through `src/telco_churn/config.py` so that scripts do not need to hard-code the local machine's absolute path.

Primary module:

- `src/telco_churn/ingestion/load_raw_data.py`

### 3.2 Data validation and profiling

The validation module profiles the source data and reports basic quality checks, including record counts, missing values, duplicate records, identifier uniqueness, and data types. The purpose is to identify source-data issues before business analysis and modelling.

Primary module:

- `src/telco_churn/validation/profile_data.py`

### 3.3 Data transformation with dbt

The dbt project transforms the ingested data into an analytical Star Schema in SQLite. The modelling layer used by the business analytics and machine-learning queries includes:

- `fact_customer_churn`
- `dim_customer`
- `dim_contract_billing`
- `dim_services`

The fact table provides the customer-level churn record. Dimension tables provide customer characteristics, contract and billing attributes, and subscribed services. Joins used for customer-level outputs should preserve the fact-table grain: one record per customer.

### 3.4 Business analytics and BI exports

SQL files in `analytics/` calculate descriptive business metrics and customer segment summaries:

- `analytics/01_contract_payment_risk.sql`
- `analytics/02_tenure_cohort_curve.sql`
- `analytics/03_service_bundle_retention.sql`
- `analytics/bi_master_view.sql`

The Python exporter in `src/telco_churn/analytics/export_bi_marts.py` executes these queries and writes CSV outputs to the configured BI export directory. The customer-level master view is checked for missing IDs, duplicate customer IDs, invalid churn labels, and row-count consistency with the fact table.

The exported datasets are intended for downstream reporting and visualization, including Tableau.

## 4. Business analytics methodology

The business analytics stage uses descriptive SQL queries to compare observed churn across customer characteristics and service or billing segments.

The analysis includes:

- Contract type and payment method combinations.
- Customer tenure groups.
- Internet and service-bundle combinations.
- Customer-level attributes in the BI master view.

Segment churn rates are calculated as the number of churned customers divided by the number of customers in the segment. Segment sizes should be shown alongside rates so that small groups are not interpreted without context.

### Revenue terminology

Where monthly charges for churned customers are summed, the metric is described as **churn-associated MRR**. It is the sum of the snapshot monthly charges associated with customers whose recorded churn label is positive.

It is not a directly observed accounting measure of revenue lost in a particular calendar month. An annualized figure calculated as 12 times churn-associated MRR is a snapshot run-rate equivalent, not realized annual revenue loss or a forecast of future losses.

### Interpretation of findings

The analysis is observational. Differences in churn rates across contracts, payment methods, tenure groups, or service bundles indicate associations in the supplied data. They do not establish that a particular service, payment method, or customer characteristic caused churn.

The customer groups may overlap. Segment-level customer counts and financial metrics must not be added together to estimate a unique total unless the overlap has been explicitly removed.

## 5. Machine-learning dataset and target

The machine-learning dataset is created by `analytics/ml/customer_training_dataset.sql`. It selects one row per customer from the dbt Star Schema by joining the fact table to the customer, contract/billing, and service dimensions.

The target is `churn_flag`.

### 5.1 Model features

The primary model uses 19 input features.

**Numeric features**

- `tenure_months`
- `monthly_charges`

**Categorical features**

- `gender`
- `is_senior_citizen`
- `has_partner`
- `has_dependents`
- `contract_type`
- `payment_method`
- `paperless_billing`
- `phone_service`
- `multiple_lines`
- `internet_service`
- `online_security`
- `online_backup`
- `device_protection`
- `tech_support`
- `streaming_tv`
- `streaming_movies`

`customer_id` is retained for record identification and validation, but is not a model feature. `total_charges` is available in the source query but is excluded from the primary model because it is cumulative and closely related to the customer's observed subscription duration. This is a modelling choice intended to reduce reliance on a variable that can encode exposure time in the snapshot.

### 5.2 Missing-value handling and preprocessing

Preprocessing is included in a scikit-learn pipeline so that transformations are fitted on the training data and then applied consistently to validation, test, and future scoring data.

The current preprocessing design is:

- Numeric features: median imputation with missingness indicators, followed by standard scaling.
- Categorical features: most-frequent-value imputation, followed by one-hot encoding.
- Unseen categories at scoring time: ignored by the one-hot encoder rather than causing a scoring failure.

The preprocessing pipeline is saved together with the selected classifier. This ensures that future input data is processed in the same way as the training data.

## 6. Train, validation, and test design

The data is divided using stratified sampling into three mutually exclusive partitions:

| Partition | Share | Purpose |
|---|---:|---|
| Training | 70% | Fit candidate models and the selected model pipeline. |
| Validation | 15% | Select the classification threshold. |
| Test | 15% | Evaluate the selected model once on held-out records. |

The split uses `random_state=42` and stratification on `churn_flag` to preserve the approximate churn proportion across partitions.

The split is recreated with the same seed in the evaluation script. The source database and modelling query must therefore remain unchanged between training and evaluation for the test partition to remain identical.

## 7. Model training and selection

The training workflow is implemented in `src/telco_churn/ml/train.py`. It compares a baseline and three candidate classification algorithms:

- Dummy Classifier using the prior class distribution.
- Logistic Regression with class weighting.
- Random Forest with balanced-sub-sample class weighting.
- HistGradientBoosting Classifier.

Five-fold stratified cross-validation is conducted on the training partition. The primary model-selection metric is **Average Precision (area under the precision-recall curve, PR-AUC)**. ROC-AUC, precision, recall, and F1 are also recorded to provide additional context.

The model with the highest mean cross-validation Average Precision is selected. ROC-AUC is used as a secondary sort criterion when comparing equal primary scores. The selected preprocessing-and-classifier pipeline is fitted on the training partition and saved for later reuse.

In this project, Logistic Regression is the selected model when it achieves the highest cross-validation Average Precision in the recorded model comparison. The actual comparison output should be used as the source of the final reported model-selection result.

## 8. Classification threshold selection

The classifier produces a churn probability for each record. A threshold is required to convert the probability into a binary class:

- Probability at or above the threshold: predicted class `1`.
- Probability below the threshold: predicted class `0`.

The training script evaluates thresholds from 0.10 to 0.90 in increments of 0.01 using the validation partition. It selects the threshold that maximizes the F1-score. Where F1 values are tied, the selection favours higher recall, followed by higher precision.

The selected threshold is stored in the model metadata and reused by the evaluation and scoring scripts.

This threshold is an empirical classification choice based on F1. It is not necessarily the threshold that maximizes business profit or minimizes retention campaign costs. A business-specific threshold would require information about intervention costs, customer value, and the effectiveness of retention actions.

## 9. Model evaluation

The evaluation workflow is implemented in `src/telco_churn/ml/evaluate.py`. It loads the saved model and metadata and evaluates the model on the held-out test partition using the same split seed and method as training.

The test partition is not used to select the model or classification threshold. The evaluation reports:

- Accuracy.
- Balanced accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Average Precision (PR-AUC).
- Brier score.

The evaluation also produces a classification report, confusion matrix, ROC curve, precision-recall curve, and test prediction records without customer identifiers.

The test results estimate retrospective classification performance on held-out records from the same source snapshot. They do not establish performance on future periods, new populations, or a future 30-day churn outcome.

## 10. Model explainability

The explainability workflow is implemented in `src/telco_churn/ml/explain.py` and uses SHAP to examine how the selected model's input features contribute to its predictions.

The analysis is intended to provide:

- **Global explanation:** an overview of which features have the greatest influence on model outputs across the records analysed.
- **Local explanation:** an explanation of how feature contributions combine for an individual customer prediction.

SHAP values describe contributions to the model output under the chosen SHAP method and background data. They do not show that a feature causes churn. Explanations should be interpreted alongside the observational nature and limitations of the dataset.

The generated SHAP figures and any accompanying outputs are stored in the configured ML figures or output directories.

## 11. Reusable scoring interface

The reusable scoring interface is implemented in `src/telco_churn/ml/score_new_customers.py`.

It is designed to accept a separate CSV file containing the same 19 feature columns used during training. The CSV may also contain `customer_id` to preserve customer-level identification. It does not require `churn_flag` because the actual outcome is not needed to generate a model score.

The scoring script:

1. Validates that the required feature columns are present.
2. Loads the saved preprocessing-and-classifier pipeline.
3. Loads the associated classification threshold from model metadata.
4. Calculates a churn score for each input record.
5. Assigns a binary predicted class using the saved threshold.
6. Exports a customer-level CSV with the score, predicted class, model version, and threshold.

The scoring interface is prepared for future use. No genuinely new customer dataset is currently available, so no claim is made that new customers have been scored. When compatible data becomes available, the input data should be checked for feature definitions, units, category meanings, data quality, and population differences before the scores are used.

A successful scoring run demonstrates that the saved pipeline can process an input file. It does not, by itself, validate prospective predictive performance.

## 12. Reproducibility and testing

The project centralizes paths in `src/telco_churn/config.py` and uses a `.project-root` marker to locate the project root. Python modules are run from the project root with `PYTHONPATH=src`.

The train/validation/test split uses a fixed random seed (`42`) to make the partitions reproducible when the underlying data and row ordering are unchanged.

The project should maintain a dependency record for the Python environment and automated tests for:

- Database existence, required tables, and customer-level uniqueness.
- Modelling dataset structure and target validity.
- Non-overlapping train, validation, and test partitions.
- Saved model and metadata availability.
- Model probability output and the reusable scoring interface.

The project is considered reproducible only when the documented commands, dependencies, database inputs, dbt models, and code versions are available and the relevant checks pass. A passing unit or integration test does not replace an independent evaluation on later-period customer data.

## 13. Limitations and next steps

The main limitations are:

1. **No dated point-in-time snapshots:** the data does not support a defined future churn prediction horizon such as 30 days.
2. **Retrospective target:** the model classifies the recorded churn label in the supplied snapshot; it is not validated as a prospective churn model.
3. **Observational analysis:** business segment differences and SHAP contributions are associations or model explanations, not causal effects.
4. **Snapshot features:** tenure and other customer attributes reflect the observed snapshot. Their availability and meaning must be checked when scoring data collected at a different point in time.
5. **Threshold selection:** the current threshold maximizes validation F1, rather than optimizing a business cost or profit function.
6. **Population and data drift:** future customer populations, product offerings, billing processes, and churn patterns may differ from the historical dataset.
7. **No intervention experiment:** the analysis does not establish that contacting customers flagged by the model will reduce churn.

The most valuable future improvement is to obtain dated customer snapshots and subsequent churn outcomes. A prospective modelling design could then define an observation date, a feature look-back period, and a fixed outcome window (for example, churn within the following 30 days). The model could be trained on earlier cohorts and evaluated on later cohorts, with performance and calibration monitored over time.

Until such data is available, the current project should be presented as an end-to-end descriptive analytics and retrospective churn-classification project, with a reusable scoring interface prepared for future compatible customer data.
