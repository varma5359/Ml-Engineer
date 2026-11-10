# Day 80: Capstone — Baseline Model and Initial Experiments

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Baseline model

## 1. Learning objectives
- Explain why a baseline is needed.
- Build a simple majority-class and logistic regression baseline.
- Evaluate predictions using classification metrics.
- Compare model results with a simple reference.
- Record reproducible initial experiment details.

## 2. What is a baseline model?
A baseline is a simple reference used to judge whether more complex models add value.

For churn classification, compare:
1. DummyClassifier: a simple reference strategy.
2. LogisticRegression: a straightforward learned classifier using the preprocessing pipeline.

A baseline is not expected to be the final production model. It provides a minimum reference and helps reveal data or evaluation problems.

## 3. Establish a dummy baseline
Continue with the train/test split and preprocessing pipeline from Day 79.

```python
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

dummy_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])

dummy_model.fit(X_train, y_train)
dummy_predictions = dummy_model.predict(X_test)
```

The dummy classifier predicts the most frequent training class. Its accuracy may look deceptively strong when churn is rare.

## 4. Train logistic regression baseline
```python
from sklearn.linear_model import LogisticRegression

baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight=None))
])

baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)
baseline_probabilities = baseline_model.predict_proba(X_test)[:, 1]
```

Use the same training split and preprocessing design for a fair comparison. If your positive class is not encoded as 1 or class order differs, inspect `classes_` before selecting a probability column.

## 5. Evaluate both models
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def classification_summary(y_true, y_pred, y_score=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    if y_score is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_score)
    return results

print("Dummy:", classification_summary(y_test, dummy_predictions))
print("Logistic:", classification_summary(
    y_test, baseline_predictions, baseline_probabilities
))
```

ROC-AUC requires both classes in the evaluation data. For strongly imbalanced data, also consider PR-AUC and precision/recall at an operational threshold.

## 6. Inspect the confusion matrix
```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_test, baseline_predictions,
    display_labels=["No churn", "Churn"]
)
plt.title("Baseline Logistic Regression")
plt.tight_layout()
plt.show()
```

Interpret:
- True positive: churn correctly flagged.
- False positive: non-churn customer flagged.
- True negative: non-churn correctly left unflagged.
- False negative: churner missed.

## 7. Create a baseline comparison table
```python
import pandas as pd

comparison = pd.DataFrame([
    {"model": "Dummy", **classification_summary(
        y_test, dummy_predictions
    )},
    {"model": "Logistic Regression", **classification_summary(
        y_test, baseline_predictions, baseline_probabilities
    )}
])
print(comparison.round(3))
```

Use consistent data, positive-label definition and metrics. Record any metric that cannot be calculated and why.

## 8. Initial experiment record
For every run, record:
- Dataset version and split method.
- Feature list and target mapping.
- Preprocessing configuration.
- Model and hyperparameters.
- Random seed.
- Metrics and confusion matrix.
- Software versions.
- Known limitations.

This record becomes the foundation for later MLflow tracking and reproducibility.

## 9. Approved deliverable — Baseline model

**Objective:** Train a simple reference model for the churn problem and document its initial performance.

### Tasks
1. Reuse the approved Day 79 split and pipeline.
2. Train a DummyClassifier.
3. Train logistic regression.
4. Calculate accuracy, precision, recall, F1 and ROC-AUC where valid.
5. Display confusion matrices.
6. Compare both models in a table.
7. Explain the effect of class imbalance.
8. Record dataset version, split, parameters and limitations.

**Suggested structure**
```text
customer_churn_capstone/
├── src/baseline.py
├── reports/baseline_report.md
├── artifacts/
└── README.md
```

**Validation:** Ensure both models use the same split. Confirm positive class and probability mapping. Do not repeatedly tune against the test set; use validation data for model development.

## 10. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for choosing a meaningful baseline." |
| Plan | "Break baseline training and comparison into steps." |
| Example | "Show a small DummyClassifier comparison example." |
| Guided implementation | "Guide me through evaluating logistic regression one metric at a time." |
| Explanation | "Explain what false negatives mean in a churn use case." |
| Debugging | "Help me debug this metric error without replacing my notebook." |
| Review | "Review my baseline report for fair comparison and metric interpretation." |

## 11. Daily quiz

**Q1. Why train a DummyClassifier?**  
A. To guarantee best performance  
B. To establish a simple reference  
C. To deploy an API  
D. To encode categories  
**Correct: B.** It shows how a simple strategy performs.

**Q2. What does a false negative mean in churn prediction?**  
A. A non-churner flagged as churn  
B. A churner not identified  
C. A churner correctly identified  
D. A model artifact missing  
**Correct: B.** The model misses a customer who churns.

**Q3. Which metric is the harmonic mean of precision and recall?**  
A. F1  
B. Accuracy  
C. RMSE  
D. R²  
**Correct: A.** F1 combines precision and recall.

**Q4. Why should candidate models use the same evaluation split?**  
A. To make the code shorter  
B. To make comparisons fairer  
C. To remove class imbalance  
D. To eliminate preprocessing  
**Correct: B.** Different data splits can confound performance comparisons.

**Q5. Which data should guide repeated model tuning?**  
A. The untouched final test set  
B. A validation set  
C. Production secrets  
D. The target column alone  
**Correct: B.** Preserve the test set for final evaluation.

## 12. Interview preparation

### Q1. Why do you need a baseline model?
**Answer:** It provides a simple performance reference. If a complex model cannot improve meaningfully over a baseline, the additional complexity may not be justified. A baseline also helps detect target, data and evaluation problems.

**Follow-up:** Which baseline would you choose for an imbalanced classification problem?

### Q2. Explain the confusion matrix for churn.
**Answer:** It counts true positives, false positives, true negatives and false negatives. In churn, false negatives are customers who churn but were not flagged; false positives are customers flagged who do not churn. Their business costs help determine the operating threshold.

**Follow-up:** Which error is more expensive for your stakeholder, and how would you verify it?

## 13. Completion checklist
- [ ] Train dummy and logistic baselines.
- [ ] Calculate classification metrics.
- [ ] Inspect confusion matrix.
- [ ] Compare models on the same split.
- [ ] Record experiment metadata.
- [ ] Complete Baseline model.

**Next lesson:** Day 81 — Capstone: Candidate model training.
