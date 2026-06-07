# Training results

- best model: Logistic Regression
- frozen threshold: 0.4971
- test precision (churn): 0.5008
- test recall (churn): 0.7995
- test f1 (churn): 0.6159
- test roc_auc: 0.8410

## Confusion matrix (test set)

| | Predicted No Churn | Predicted Churn |
|---|---|---|
| Actual No Churn | 737 | 298 |
| Actual Churn | 75 | 299 |

## Business interpretation

At threshold **0.50**, the model flags about **424 customers per 1,000**, of whom roughly **50%** are real churners. On the held-out test sample (1,409 customers), that means **597 marketing calls**, catching **299** actual churners while missing the rest.

Because a missed churner costs about 5x more than calling a loyal customer, we prioritised recall on validation data and then chose the highest precision among thresholds with recall >= 0.80.
