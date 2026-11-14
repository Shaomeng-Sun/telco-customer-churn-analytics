# Churn Model Evaluation

Evaluation on the held-out test partition. The threshold was selected using validation data only.

- Model: **logistic_regression**
- Threshold: **0.54**
- Test records: **1,057**

## Metrics

| Metric | Test result |
|---|---:|
| Accuracy | 0.7550 |
| Balanced Accuracy | 0.7548 |
| Precision | 0.5274 |
| Recall | 0.7544 |
| F1 | 0.6208 |
| Roc Auc | 0.8324 |
| Average Precision Pr Auc | 0.6398 |
| Brier Score | 0.1737 |

## Classification report

```text
              precision    recall  f1-score   support

 Not churned       0.89      0.76      0.82       776
     Churned       0.53      0.75      0.62       281

    accuracy                           0.75      1057
   macro avg       0.71      0.75      0.72      1057
weighted avg       0.80      0.75      0.77      1057

```

> **Scope limitation:** These metrics describe retrospective classification of the supplied customer snapshot. They do not establish performance for a future 30-day churn prediction task.