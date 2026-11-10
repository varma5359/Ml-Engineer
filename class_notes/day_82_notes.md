# Day 82: Capstone — Evaluation and Model Selection

**Pathway:** Machine Learning Engineer  
**Phase:** 8 — End-to-End ML Capstone  
**Level:** Intermediate  
**Estimated time:** 2–3 hours  
**Capstone:** Customer Churn Prediction and ML Deployment System  
**Approved mini-project / deliverable:** Evaluation report

## 1. Learning objectives
- Evaluate a selected candidate on an appropriate held-out dataset.
- Interpret confusion matrix and classification metrics.
- Compare ROC and precision-recall behavior.
- Examine performance across meaningful customer segments.
- Write a model selection report with limitations.

## 2. Evaluation is more than one score
A useful evaluation answers:
- How well does the model rank customers by risk?
- What happens at the chosen decision threshold?
- How many churners are missed?
- How many non-churners are flagged?
- Does performance differ across customer segments?
- Does the model satisfy business review capacity?

Use validation data for model selection. Once the pipeline and decision are fixed, evaluate the chosen model on the untouched final test set.

## 3. Load the selected candidate
Use the candidate chosen from validation experiments:
```python
selected_model = trained_models["random_forest"]  # Example only
```

The actual selection must be justified by validation evidence, not by copying this example name.

## 4. Calculate final metrics
```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

final_predictions = selected_model.predict(X_test)
final_scores = selected_model.predict_proba(X_test)[:, 1]

print(classification_report(
    y_test, final_predictions,
    target_names=["No churn", "Churn"],
    zero_division=0
))
print("ROC-AUC:", roc_auc_score(y_test, final_scores))
print("PR-AUC:", average_precision_score(y_test, final_scores))
print(confusion_matrix(y_test, final_predictions))
```

Confirm that class label 1 represents churn and that probability column 1 corresponds to that class. Check `selected_model.classes_` if needed.

## 5. Visualize ROC and precision-recall curves
```python
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

RocCurveDisplay.from_predictions(y_test, final_scores)
plt.title("ROC Curve — Final Candidate")
plt.show()

PrecisionRecallDisplay.from_predictions(y_test, final_scores)
plt.title("Precision-Recall Curve — Final Candidate")
plt.show()
```

ROC curves describe true-positive versus false-positive rates across thresholds. Precision-recall curves show the precision/recall trade-off and can be especially useful for imbalanced positive classes.

## 6. Select an operating threshold
The default threshold for many classifiers is 0.5, but it is not automatically appropriate.

Use validation scores to choose a threshold that matches the agreed business constraints. For example, maximize recall subject to a minimum precision or a review-capacity limit.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

threshold = 0.35  # Illustrative; select using validation data.
threshold_predictions = (validation_scores >= threshold).astype(int)

print("Precision:", precision_score(
    y_validation, threshold_predictions, zero_division=0
))
print("Recall:", recall_score(
    y_validation, threshold_predictions, zero_division=0
))
```

Do not select the threshold using the final test labels. Record the chosen threshold as part of the model configuration.

## 7. Segment evaluation
Compare performance across meaningful, sufficiently large segments such as contract type or tenure bands. Use only appropriate, permitted attributes and follow governance policy.

```python
evaluation_frame = X_test.copy()
evaluation_frame["actual"] = y_test.to_numpy()
evaluation_frame["predicted"] = final_predictions

for segment in ["Contract", "InternetService"]:
    if segment in evaluation_frame.columns:
        print("\\n", segment)
        print(evaluation_frame.groupby(segment).apply(
            lambda group: pd.Series({
                "rows": len(group),
                "recall": recall_score(
                    group["actual"], group["predicted"], zero_division=0
                ),
                "precision": precision_score(
                    group["actual"], group["predicted"], zero_division=0
                )
            })
        ))
```

Small groups can produce unstable estimates. Report counts and uncertainty; do not draw strong conclusions from tiny samples.

## 8. Model selection decision
A decision report should explain:
- Primary selection metric and why.
- Threshold and business capacity.
- Overall and segment performance.
- Baseline comparison.
- Runtime, complexity and artifact size.
- Known limitations and conditions for deployment.

## 9. Approved deliverable — Evaluation report

**Objective:** Produce a final evaluation report for the selected churn candidate.

### Tasks
1. Freeze candidate selection based on validation evidence.
2. Evaluate the chosen pipeline on the final test set.
3. Report confusion matrix, precision, recall, F1, ROC-AUC and PR-AUC where valid.
4. Show ROC and precision-recall curves.
5. Explain threshold selection from validation data.
6. Examine performance across suitable segments.
7. Compare against the Day 80 baseline.
8. Document limitations and deployment recommendation.

**Suggested structure**
```text
customer_churn_capstone/
├── reports/evaluation_report.md
├── reports/figures/
└── artifacts/selected_model/
```

**Validation:** Confirm test data was not used to tune the model or threshold. Verify positive-label mapping, report segment sample sizes and distinguish measured facts from assumptions.

## 10. AI-assisted implementation guide
| Level | Prompt |
|---|---|
| Hint | "Give me one hint for interpreting my confusion matrix." |
| Plan | "Outline an evaluation report for a binary churn classifier." |
| Example | "Show a compact comparison of precision and recall at two thresholds." |
| Guided implementation | "Help me add the precision-recall curve to my evaluation notebook." |
| Explanation | "Explain why PR-AUC and ROC-AUC can tell different stories." |
| Debugging | "Check why my class labels and display labels appear reversed." |
| Review | "Review my evaluation report for test leakage and unsupported claims." |

## 11. Daily quiz

**Q1. Which data should be used for the final performance estimate after model selection?**  
A. Training set  
B. Untouched test set  
C. All data before splitting  
D. Only one positive row  
**Correct: B.** The test set is reserved for the final estimate.

**Q2. What does a false positive represent?**  
A. Churner correctly flagged  
B. Non-churner incorrectly flagged  
C. Churner missed  
D. Non-churner correctly rejected  
**Correct: B.** The model flags a customer who does not churn.

**Q3. Why examine a precision-recall curve?**  
A. To inspect precision/recall trade-offs across thresholds  
B. To calculate file size  
C. To measure API latency  
D. To encode categories  
**Correct: A.** It shows the trade-off as the threshold changes.

**Q4. When should a classification threshold be selected?**  
A. On final test labels repeatedly  
B. On validation evidence and business constraints  
C. By selecting 0.5 in every problem  
D. After deployment without review  
**Correct: B.** Threshold decisions belong in model selection.

**Q5. Why report segment sample sizes?**  
A. To make the report longer  
B. To show the stability and context of segment metrics  
C. To hide low recall  
D. To avoid using labels  
**Correct: B.** Small groups can produce unstable estimates.

## 12. Interview preparation

### Q1. Explain the difference between ROC-AUC and PR-AUC.
**Answer:** ROC-AUC summarizes ranking across true-positive and false-positive rates. PR-AUC summarizes precision and recall across thresholds. PR-AUC is often more revealing when the positive class is rare, but metric choice should match the use case and prevalence.

**Follow-up:** Why can a high ROC-AUC coexist with low precision at a particular threshold?

### Q2. How do you choose a classification threshold?
**Answer:** Use validation scores and the business costs of false positives and false negatives. Evaluate precision, recall, flagged volume and segment behavior across thresholds. Document the selected threshold and keep the final test set for the final estimate.

**Follow-up:** What if review capacity changes after deployment?

## 13. Completion checklist
- [ ] Evaluate the selected candidate on untouched test data.
- [ ] Report confusion matrix and classification metrics.
- [ ] Plot ROC and precision-recall curves.
- [ ] Justify threshold from validation data.
- [ ] Inspect segment metrics and sample sizes.
- [ ] Complete Evaluation report.

**Next lesson:** Day 83 — Capstone: Tuning and validation.
