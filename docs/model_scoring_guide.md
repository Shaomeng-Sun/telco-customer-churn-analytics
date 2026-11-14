# Customer Churn Model – Scoring Guide

## 1. Purpose

This guide explains how to use the trained Logistic Regression pipeline to generate churn scores for customer records supplied as a CSV file.

The scoring workflow reuses the saved model and its associated classification threshold. It does not retrain the model.

## 2. Required input

The input CSV must contain the following 19 model features:

**Numeric features**
- tenure_months
- monthly_charges

**Categorical features**
- gender
- is_senior_citizen
- has_partner
- has_dependents
- contract_type
- payment_method
- paperless_billing
- phone_service
- multiple_lines
- internet_service
- online_security
- online_backup
- device_protection
- tech_support
- streaming_tv
- streaming_movies

The CSV may also contain customer_id to identify individual records.

The input must not contain churn_flag because the actual outcome is not required for scoring.

All feature definitions, units and category meanings must be consistent with the training dataset.

## 3. Scoring command

Place the customer CSV in the project data directory and execute:

```bash
PYTHONPATH=src python -m telco_churn.ml.score_new_customers \
    --input data/incoming/customer_snapshot.csv \
    --data data/ml/customer_scores.csv
```

## 4. Output

The scoring script produces a CSV containing:

- customer_id, when supplied
- model_churn_score
- predicted_churn_class_at_validation_threshold
- model_version
- classification_threshold

The saved classification threshold from training is applied to all input records.

## 5. Model limitations

The current model performs retrospective classification of the recorded churn label in the historical customer snapshot.

It has not been validated as a prospective 30-day churn prediction model.

Before using scores for actual customer retention decisions, performance should be validated using appropriately dated customer snapshots and subsequently observed churn outcomes.

## 6. Model artifacts

The scoring workflow uses the saved model and metadata produced during Step 5.

The model must not be retrained simply to score another compatible customer dataset.