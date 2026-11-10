# Day 46: ROC Curve, AUC and PR Curve

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain prediction scores and decision thresholds.
- Understand ROC curve axes and True Positive Rate / False Positive Rate.
- Explain ROC-AUC and its limitations.
- Generate ROC curves using scikit-learn.
- Understand Precision-Recall curves and Average Precision.
- Compare ROC and PR analysis for imbalanced classification.
- Select a threshold using validation evidence and business constraints.
- Explain ROC-AUC and PR metrics in interviews.

## 2. Why do we need ROC and Precision-Recall curves?

A classifier often provides a score for each observation. A threshold converts that score into a final class.

Example:
- Score 0.92: model considers the observation strongly positive.
- Score 0.18: model considers the observation less likely to be positive.
- Threshold 0.50: scores at or above 0.50 are predicted positive.

The best threshold depends on the business problem. ROC and Precision-Recall curves show how model performance changes across different thresholds.

They help us:
- Compare ranking behaviour across thresholds.
- Understand the trade-off between detecting positives and creating false alerts.
- Select a possible operating threshold.
- Compare candidate classifiers before deployment.

## 3. Prediction scores and predicted labels

| Output | Meaning |
|---|---|
| Prediction score | A model's score for an observation |
| Probability estimate | A score intended to represent probability, if calibrated |
| Decision threshold | A cutoff used to convert scores into class labels |
| Predicted label | Final class such as 0 or 1 |

Important:
- A model score is not automatically a well-calibrated probability.
- ROC and PR curves use scores across thresholds, not just one set of hard labels.
- The chosen threshold determines the deployed operating point.

## 4. Understand ROC curve

**ROC** means Receiver Operating Characteristic.

A ROC curve plots:
- **True Positive Rate (TPR)** on the vertical axis.
- **False Positive Rate (FPR)** on the horizontal axis.

TPR is also called recall or sensitivity.

FPR describes the share of actual negative observations incorrectly predicted as positive.

The ROC curve shows how TPR and FPR change as the decision threshold changes.

### 4.1 Interpret ROC curve behaviour

- A curve closer to the top-left generally indicates better separation between positive and negative classes.
- A diagonal line represents random ranking in the usual ROC plot.
- A model curve below the diagonal can indicate reversed ranking; check score direction and the positive class.
- ROC behaviour should be assessed on representative evaluation data.

## 5. Understand AUC

**AUC** means Area Under the Curve. When we say ROC-AUC, we mean the area under the ROC curve.

AUC is commonly interpreted as the model's ability to rank a randomly selected positive observation above a randomly selected negative observation.

General interpretation:
- AUC near 1: strong ranking separation.
- AUC near 0.5: ranking is similar to random.
- AUC below 0.5: ranking may be reversed or worse than random.

AUC is not:
- Classification accuracy.
- A guarantee that one threshold is suitable.
- A probability that a particular prediction is correct.
- Proof that the model is useful for the business.

**Interview point:** ROC-AUC summarizes ranking performance across thresholds. It does not tell the team which threshold to deploy.

## 6. Create a small binary classification example

### 6.1 Import libraries

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Libraries ready")
~~~

**Expected output**
~~~text
Libraries ready
~~~

**Explanation**
- NumPy stores sample labels and scores.
- Pandas helps create a readable table.
- Matplotlib draws the curves.

### 6.2 Create actual labels

~~~python
y_true = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])

print(y_true)
~~~

**Expected output**
A NumPy array containing five zeros followed by five ones.

**Explanation**
- 0 represents the negative class.
- 1 represents the positive class.
- We know the correct labels because this is a teaching example.

### 6.3 Create model scores

~~~python
y_scores = np.array([
    0.10, 0.20, 0.35, 0.40, 0.60,
    0.45, 0.55, 0.70, 0.80, 0.95
])

print(y_scores)
~~~

**Expected output**
A NumPy array of ten scores between 0 and 1.

**Explanation**
These are illustrative scores. The model ranks observations by how strongly they appear to belong to the positive class.

### 6.4 Inspect actual labels and scores together

~~~python
score_table = pd.DataFrame({
    "actual": y_true,
    "score": y_scores
})

print(score_table)
~~~

**Expected output**
A ten-row table with actual class and score.

**Explanation**
The table helps us see that some positive observations receive lower scores than some negative observations. This overlap affects curve performance.

## 7. Generate the ROC curve

### 7.1 Import ROC functions

~~~python
from sklearn.metrics import roc_curve, roc_auc_score

print("ROC functions imported")
~~~

**Expected output**
~~~text
ROC functions imported
~~~

### 7.2 Calculate ROC curve values

~~~python
fpr, tpr, thresholds = roc_curve(
    y_true,
    y_scores,
    pos_label=1
)

print("FPR:", fpr)
print("TPR:", tpr)
print("Thresholds:", thresholds)
~~~

**Expected output**
Three arrays containing false positive rates, true positive rates and the corresponding score thresholds.

**Explanation**
- fpr contains horizontal-axis values.
- tpr contains vertical-axis values.
- thresholds contains the cutoffs used to generate each point.
- The first threshold may be infinite to represent a point where no observations are predicted positive. This is expected in current scikit-learn versions.

### 7.3 Display ROC points as a table

~~~python
roc_table = pd.DataFrame({
    "threshold": thresholds,
    "false_positive_rate": fpr,
    "true_positive_rate": tpr
})

print(roc_table)
~~~

**Expected output**
A table with one row for each ROC operating point.

**Explanation**
The table allows us to inspect how changing the threshold changes false positive and true positive rates.

### 7.4 Plot the ROC curve

~~~python
plt.plot(
    fpr,
    tpr,
    marker="o",
    label="Classifier ROC"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random ranking reference"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
~~~

**Expected output**
A ROC chart with the classifier curve and a diagonal reference line.

**Explanation**
- A curve that rises toward the top-left is generally preferable.
- The diagonal is a reference for random ranking.
- The plot displays behaviour over many thresholds rather than one threshold only.

## 8. Calculate ROC-AUC

### 8.1 Calculate AUC

~~~python
roc_auc = roc_auc_score(
    y_true,
    y_scores
)

print("ROC-AUC:", roc_auc)
~~~

**Expected output**
A numeric score between 0 and 1 for this binary example.

**Explanation**
The score summarizes the ROC curve's ranking performance. It is not a probability that an individual prediction is correct.

### 8.2 Compare two score sets

~~~python
scores_model_a = np.array([
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.55, 0.65, 0.75, 0.85, 0.95
])

scores_model_b = np.array([
    0.20, 0.35, 0.45, 0.60, 0.70,
    0.40, 0.50, 0.65, 0.80, 0.90
])

auc_a = roc_auc_score(y_true, scores_model_a)
auc_b = roc_auc_score(y_true, scores_model_b)

print("Model A AUC:", auc_a)
print("Model B AUC:", auc_b)
~~~

**Expected output**
Two AUC values that can be compared to assess ranking performance.

**Explanation**
A higher AUC indicates better ranking separation on this evaluation sample. Check that both models use the same observations and positive-class definition.

## 9. Understand Precision-Recall curve

A **Precision-Recall (PR) curve** plots:
- Precision on the vertical axis.
- Recall on the horizontal axis.

It focuses on positive-class performance.

The curve shows how precision and recall change as the decision threshold changes.

PR curves are particularly useful when:
- The positive class is rare.
- Correctly identifying positive cases is the main goal.
- False positive alerts matter.
- The business needs to understand positive prediction quality at different recall levels.

## 10. Generate a Precision-Recall curve

### 10.1 Import precision_recall_curve

~~~python
from sklearn.metrics import precision_recall_curve

print("Precision-recall function imported")
~~~

**Expected output**
~~~text
Precision-recall function imported
~~~

### 10.2 Calculate precision and recall arrays

~~~python
precision_values, recall_values, pr_thresholds = precision_recall_curve(
    y_true,
    y_scores,
    pos_label=1
)

print("Precision:", precision_values)
print("Recall:", recall_values)
print("Thresholds:", pr_thresholds)
~~~

**Expected output**
Three arrays containing precision values, recall values and decision thresholds.

**Explanation**
The precision and recall arrays have one more value than the threshold array in scikit-learn. The final precision-recall point has no corresponding threshold.

### 10.3 Create a PR table

~~~python
pr_table = pd.DataFrame({
    "threshold": pr_thresholds,
    "precision": precision_values[:-1],
    "recall": recall_values[:-1]
})

print(pr_table)
~~~

**Expected output**
A table connecting each threshold with its precision and recall values.

**Explanation**
We use the precision and recall arrays without their final endpoint because they contain one extra value compared with the threshold array.

### 10.4 Plot the PR curve

~~~python
plt.plot(
    recall_values,
    precision_values,
    marker="."
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
~~~

**Expected output**
A chart showing precision against recall for different thresholds.

**Explanation**
The curve helps a team understand how much precision it may give up to capture more actual positive cases.

## 11. Calculate PR-AUC and Average Precision

There are related but not identical ways to summarize a PR curve.

- **Average Precision (AP):** A summary calculated using precision values across recall increments.
- **PR-AUC:** Often used informally for area under a precision-recall curve; the exact numerical method should be stated because integration conventions can differ.

For many scikit-learn workflows, average_precision_score is a clear and commonly reported PR summary.

### 11.1 Calculate Average Precision

~~~python
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(
    y_true,
    y_scores
)

print("Average Precision:", average_precision)
~~~

**Expected output**
A numeric average precision value between 0 and 1.

**Explanation**
AP summarizes the precision-recall ranking behaviour. It is not the same thing as classification accuracy.

### 11.2 Calculate trapezoidal PR area for comparison

~~~python
from sklearn.metrics import auc

pr_trapezoid_area = auc(
    recall_values,
    precision_values
)

print("Trapezoidal PR area:", pr_trapezoid_area)
~~~

**Expected output**
A numeric area value.

**Explanation**
This uses trapezoidal integration of the plotted PR points. It may differ from Average Precision. When reporting results, name the exact metric and method rather than using AP and PR-AUC interchangeably.

## 12. ROC versus Precision-Recall curve

| Aspect | ROC curve | Precision-Recall curve |
|---|---|---|
| X-axis | False Positive Rate | Recall |
| Y-axis | True Positive Rate | Precision |
| Focus | Positive detection versus false positive rate | Positive prediction quality |
| Common summary | ROC-AUC | Average Precision or stated PR-AUC |
| Useful when | Overall ranking separation matters | Positive class is rare or central |
| Main caution | Can look strong on highly imbalanced data | Baseline precision depends on positive prevalence |

Both curves should be interpreted with the class distribution and business costs.

## 13. Why ROC-AUC can look strong on imbalanced data

When negative cases are extremely common, even a relatively small false-positive rate can create many false alerts.

Example:
- 100,000 total observations.
- 1,000 positive cases.
- 99,000 negative cases.
- A false positive rate of 1% produces 990 false positives.

The false positive rate looks small, but the number of false alerts is almost as large as the number of actual positive cases.

A PR curve makes the precision impact of these alerts more visible.

**Interview point:** For rare positive events, report ROC-AUC alongside PR-AUC or Average Precision, positive-class precision and recall, and threshold-specific counts.

## 14. Compare curves using an imbalanced example

### 14.1 Create imbalanced labels

~~~python
y_rare = np.array([0] * 95 + [1] * 5)

print("Negative:", (y_rare == 0).sum())
print("Positive:", (y_rare == 1).sum())
~~~

**Expected output**
~~~text
Negative: 95
Positive: 5
~~~

### 14.2 Create example scores

~~~python
rare_scores = np.array(
    [0.05 + (i % 10) * 0.02 for i in range(95)] +
    [0.45, 0.55, 0.65, 0.75, 0.90]
)

print("Score count:", len(rare_scores))
~~~

**Expected output**
~~~text
Score count: 100
~~~

**Explanation**
This is a teaching example. Some negative observations receive moderate scores, so the model may produce false alerts depending on threshold.

### 14.3 Calculate both summary metrics

~~~python
rare_roc_auc = roc_auc_score(y_rare, rare_scores)
rare_average_precision = average_precision_score(
    y_rare,
    rare_scores
)

print("ROC-AUC:", rare_roc_auc)
print("Average Precision:", rare_average_precision)
~~~

**Expected output**
Two ranking summary values.

**Explanation**
Compare both measures. The PR baseline is connected to the proportion of positive observations, so class prevalence matters when interpreting AP.

### 14.4 Plot ROC curve

~~~python
rare_fpr, rare_tpr, _ = roc_curve(y_rare, rare_scores)

plt.plot(rare_fpr, rare_tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Rare Positive Class")
plt.show()
~~~

**Expected output**
A ROC curve for the rare-positive example.

### 14.5 Plot PR curve

~~~python
rare_precision, rare_recall, _ = precision_recall_curve(
    y_rare,
    rare_scores
)

plt.plot(rare_recall, rare_precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Rare Positive Class")
plt.show()
~~~

**Expected output**
A precision-recall curve for the same scores.

**Explanation**
Use both charts to understand ranking behaviour and positive-class trade-offs.

## 15. Choose a threshold using business requirements

Do not choose a threshold just because it is 0.50 or because it gives the largest AUC. AUC evaluates ranking across thresholds; the deployed model uses one operating threshold.

A threshold selection process:
1. Define the positive class.
2. Identify the cost of false positives and false negatives.
3. Define operational constraints, such as review capacity.
4. Use validation data to compare thresholds.
5. Select an operating point that satisfies requirements.
6. Confirm performance on the untouched test set.
7. Monitor the chosen operating point after deployment.

### 15.1 Build a threshold performance table

~~~python
from sklearn.metrics import confusion_matrix, precision_score, recall_score

threshold_rows = []

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    predicted = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1]
    ).ravel()

    threshold_rows.append({
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision_score(
            y_true, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_true, predicted, zero_division=0
        )
    })

threshold_report = pd.DataFrame(threshold_rows)
print(threshold_report)
~~~

**Expected output**
A table showing confusion-matrix counts, precision and recall at each threshold.

**Explanation**
This makes the practical impact of threshold changes visible. A business team can review how many cases would be flagged and how many positive cases might be missed.

## 16. Important limitations

- ROC-AUC measures ranking, not calibration.
- A high AUC does not guarantee good predictions at the chosen threshold.
- A PR curve is sensitive to positive-class prevalence.
- AUC does not encode the business cost of errors.
- Scores from different models may not be calibrated to the same probability scale.
- AUC estimates can be uncertain when there are very few positive cases.
- Curves can change when the evaluation population changes.
- Data leakage can make curves look unrealistically strong.

## 17. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Calculate ROC using only hard labels | Use continuous scores or probabilities |
| Assume AUC is accuracy | Explain it as ranking performance |
| Treat AUC as probability of a correct individual prediction | Explain its pairwise ranking interpretation |
| Use ROC-AUC alone for a rare positive class | Also inspect PR curve and Average Precision |
| Confuse AP with trapezoidal PR area | State the exact summary metric used |
| Choose threshold from test data | Tune on validation data |
| Ignore positive-class definition | Set and document the positive label |
| Assume 0.50 is always best | Choose based on business needs |
| Ignore score calibration | Evaluate calibration separately when probabilities drive decisions |

## 18. Interview-ready explanations

**What is ROC curve?**
- A plot of true positive rate against false positive rate across thresholds.
- It shows how positive detection changes relative to false alarms.
- It helps compare ranking behaviour across thresholds.

**What is ROC-AUC?**
- The area under the ROC curve.
- It summarizes ranking separation between positive and negative observations.
- It does not choose the production threshold or guarantee calibrated probabilities.

**What is a Precision-Recall curve?**
- A plot of precision against recall across thresholds.
- It focuses on positive-class prediction quality.
- It is particularly informative when positive cases are rare.

**Why can PR curve be more informative for imbalanced classification?**
- It directly shows how many predicted positives are correct as recall changes.
- It makes false-positive impact on precision visible.
- Its baseline depends on positive-class prevalence.

**What is the difference between Average Precision and PR-AUC?**
- Average Precision is a defined summary based on precision values across recall increments.
- PR-AUC may refer to numerical integration of the plotted curve.
- The methods can produce different values, so report the exact method.

## 19. Scenario-based interview practice

### Scenario 1: ROC-AUC is high but the business sees too many false alerts
**Expected response:**
- ROC-AUC summarizes ranking across thresholds, not the chosen operating point.
- Inspect the confusion matrix, precision and alert count at the deployed threshold.
- Evaluate the PR curve and Average Precision if positives are rare.
- Select a threshold using validation data and review capacity.

### Scenario 2: Fraud is rare and ROC-AUC looks strong
**Expected response:**
- Check positive-class prevalence.
- Calculate PR curve and Average Precision.
- Inspect precision, recall and false alert counts at candidate thresholds.
- Explain the actual number of false positives, not only the false-positive rate.

### Scenario 3: Two models have similar ROC-AUC
**Expected response:**
- Compare PR behaviour if the positive class is rare.
- Compare threshold-specific precision, recall and confusion matrices.
- Check calibration if probability values are used operationally.
- Compare stability and business cost before choosing.

### Scenario 4: The team chooses the threshold that maximizes F1 on the test set
**Expected response:**
- This uses the final test set for model selection.
- Choose the threshold on validation data.
- Lock the decision and evaluate once on the untouched test set.
- Document the business objective and accepted trade-off.

## 20. Lesson completion checklist

- [ ] I can explain scores, probabilities, thresholds and predicted labels.
- [ ] I can describe ROC axes and ROC-AUC.
- [ ] I can calculate and plot a ROC curve.
- [ ] I can describe PR curve and Average Precision.
- [ ] I understand why PR curves are useful for rare positive classes.
- [ ] I can explain the difference between AP and trapezoidal PR area.
- [ ] I can compare threshold-specific precision and recall.
- [ ] I can explain why AUC does not choose a production threshold.
- [ ] I can select a threshold using validation data and business requirements.

**Next lesson:** Day 47 — Regression Evaluation: MAE, MSE, RMSE and R².
