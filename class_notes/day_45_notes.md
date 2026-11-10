# Day 45: Accuracy, Precision, Recall and F1-Score

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain accuracy, precision, recall and F1-score in plain language.
- Understand which confusion-matrix values each metric uses.
- Choose metrics based on business costs and problem requirements.
- Explain why accuracy can mislead on imbalanced datasets.
- Calculate metrics with Python and scikit-learn.
- Compare class-specific and averaged classification metrics.
- Interpret precision-recall trade-offs in real projects.
- Answer practical interview questions about metric selection.

## 2. Why do we need multiple classification metrics?

A classification model can make different kinds of mistakes. One score cannot always describe all of them.

Imagine a bank evaluates 10,000 transactions:
- 9,900 are normal.
- 100 are fraudulent.
- A model predicts every transaction as normal.

The model is correct for 9,900 transactions. Its accuracy looks very high, but it detects zero fraud cases.

This is why we study multiple metrics:
- Accuracy describes overall correctness.
- Precision describes how reliable positive predictions are.
- Recall describes how many actual positive cases are found.
- F1-score combines precision and recall into one balanced score.

**Interview point:** Never choose a metric only because it is popular. First identify the positive class, class distribution and business cost of each error.

## 3. Review the confusion matrix

Assume:
- Positive class = Fraud.
- Negative class = Normal.

| Actual / Predicted | Predicted Fraud | Predicted Normal |
|---|---:|---:|
| Actual Fraud | TP | FN |
| Actual Normal | FP | TN |

- TP: Fraud correctly detected.
- TN: Normal transaction correctly identified.
- FP: Normal transaction incorrectly flagged as fraud.
- FN: Fraud incorrectly predicted as normal.

Accuracy, precision, recall and F1-score are calculated from these four counts.

## 4. Accuracy

**Accuracy** tells us the proportion of all predictions that are correct.

Use it when:
- Classes are reasonably balanced.
- False positives and false negatives have similar impact.
- Overall correctness is a useful summary.

Be careful when:
- One class is much more common than another.
- The minority class is the most important class.
- Different error types have very different business costs.

### 4.1 Create a small example

~~~python
y_actual = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
y_predicted = [1, 0, 0, 0, 1, 1, 0, 1, 0, 0]

print("Actual:", y_actual)
print("Predicted:", y_predicted)
~~~

**Expected output**
Two lists containing ten actual and predicted labels.

**Explanation**
- 1 represents Fraud.
- 0 represents Normal.
- These labels are the same teaching example used in Day 44.

### 4.2 Calculate accuracy using scikit-learn

~~~python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_actual, y_predicted)

print("Accuracy:", accuracy)
~~~

**Expected output**
~~~text
Accuracy: 0.8
~~~

**Explanation**
The model correctly predicted 8 of the 10 observations.

### 4.3 Calculate accuracy manually

~~~python
correct = sum(
    actual == predicted
    for actual, predicted in zip(y_actual, y_predicted)
)

total = len(y_actual)
accuracy_manual = correct / total

print("Correct:", correct)
print("Total:", total)
print("Accuracy:", accuracy_manual)
~~~

**Expected output**
~~~text
Correct: 8
Total: 10
Accuracy: 0.8
~~~

**Explanation**
- zip pairs each actual label with its prediction.
- The comparison returns True when both labels match.
- sum counts the correct comparisons.
- This manual version is for understanding; use tested library functions in normal projects.

## 5. Precision

**Precision** answers:

"When the model predicts positive, how often is that prediction actually positive?"

For fraud detection:
- Of all transactions flagged as fraud, how many were truly fraudulent?
- High precision means fewer false alarms among positive predictions.

Precision is important when:
- False positive alerts are expensive.
- Human review capacity is limited.
- Blocking legitimate users causes significant harm.
- A positive prediction triggers an expensive action.

### 5.1 Calculate precision using scikit-learn

~~~python
from sklearn.metrics import precision_score

precision = precision_score(
    y_actual,
    y_predicted,
    pos_label=1
)

print("Precision:", precision)
~~~

**Expected output**
~~~text
Precision: 0.75
~~~

**Explanation**
The model predicted four transactions as fraud, and three were actually fraud. Precision describes the reliability of those positive predictions.

### 5.2 Identify false positive impact

~~~python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(
    y_actual,
    y_predicted,
    labels=[0, 1]
).ravel()

print("False positives:", fp)
print("True positives:", tp)
~~~

**Expected output**
~~~text
False positives: 1
True positives: 3
~~~

**Explanation**
A false positive is a normal transaction that the model incorrectly flagged. Reducing false positives can improve precision, but changes may also affect recall.

## 6. Recall

**Recall** answers:

"Of all actual positive cases, how many did the model successfully find?"

For fraud detection:
- Of all fraudulent transactions, how many did the model detect?
- High recall means fewer actual positive cases are missed.

Recall is important when:
- Missing a positive case is very costly.
- The goal is to identify as many positive cases as possible.
- A later human review stage can handle some false alarms.

Examples:
- Disease screening: missing a potential disease case can be serious.
- Fraud detection: undetected fraud can cause financial loss.
- Defect inspection: a defective product passing quality control can be costly.

### 6.1 Calculate recall

~~~python
from sklearn.metrics import recall_score

recall = recall_score(
    y_actual,
    y_predicted,
    pos_label=1
)

print("Recall:", recall)
~~~

**Expected output**
~~~text
Recall: 0.75
~~~

**Explanation**
There were four actual fraud cases. The model correctly identified three and missed one.

### 6.2 Inspect false negatives

~~~python
print("False negatives:", fn)
print("Actual positive cases:", tp + fn)
~~~

**Expected output**
~~~text
False negatives: 1
Actual positive cases: 4
~~~

**Explanation**
A false negative is an actual fraud case that the model predicted as normal. Reviewing false negatives can reveal missing features, data quality problems or an unsuitable threshold.

## 7. F1-score

**F1-score** combines precision and recall into one measure.

It is useful when:
- Both false positives and false negatives matter.
- We want a single summary of the precision-recall balance.
- The positive class is more important than overall accuracy.

Limitations:
- F1 does not include true negatives directly.
- A high F1-score does not automatically mean the model is suitable for deployment.
- Business costs and class-specific results still need review.

### 7.1 Calculate F1-score

~~~python
from sklearn.metrics import f1_score

f1 = f1_score(
    y_actual,
    y_predicted,
    pos_label=1
)

print("F1-score:", f1)
~~~

**Expected output**
~~~text
F1-score: 0.75
~~~

**Explanation**
The example has equal precision and recall, so the F1-score has the same value.

### 7.2 Calculate all four metrics together

~~~python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

metric_results = {
    "Accuracy": accuracy_score(y_actual, y_predicted),
    "Precision": precision_score(y_actual, y_predicted),
    "Recall": recall_score(y_actual, y_predicted),
    "F1-score": f1_score(y_actual, y_predicted)
}

print(metric_results)
~~~

**Expected output**
~~~text
{'Accuracy': 0.8, 'Precision': 0.75, 'Recall': 0.75, 'F1-score': 0.75}
~~~

**Explanation**
A metric dictionary makes the evaluation summary easy to inspect. In a real report, include the positive-class definition and test-set size.

## 8. Understand the precision-recall trade-off

A classifier often produces a score or probability. A decision threshold converts that score into a class prediction.

Changing the threshold can change:
- How many observations are predicted positive.
- False positive count.
- False negative count.
- Precision.
- Recall.

Common pattern:
- Lower threshold: more observations are flagged positive; recall often increases and precision may decrease.
- Higher threshold: fewer observations are flagged positive; precision may increase and recall may decrease.

This is a common pattern, not a guarantee for every dataset or threshold step.

**Interview point:** The threshold should be selected using validation data and business costs. Do not tune it on the final test set.

## 9. Demonstrate how threshold changes predictions

### 9.0 Import libraries for the threshold experiment

~~~python
import numpy as np
import pandas as pd

print("Threshold experiment libraries ready")
~~~

**Expected output**
~~~text
Threshold experiment libraries ready
~~~

**Explanation**
NumPy creates the sample score arrays, and Pandas displays the threshold comparison table.

### 9.1 Create actual labels and model scores

~~~python
import numpy as np

y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])

fraud_scores = np.array([
    0.10, 0.20, 0.35, 0.45, 0.55,
    0.60, 0.80, 0.30, 0.90, 0.15
])

print("Scores:", fraud_scores)
~~~

**Expected output**
A NumPy array containing ten scores between 0 and 1.

**Explanation**
These scores are illustrative model outputs. They are used to show how a threshold changes predicted labels.

### 9.2 Predict with a threshold of 0.50

~~~python
threshold = 0.50

predictions_050 = (
    fraud_scores >= threshold
).astype(int)

print(predictions_050)
~~~

**Expected output**
~~~text
[0 0 0 0 1 1 1 0 1 0]
~~~

**Explanation**
Scores at or above 0.50 are assigned to the positive class. Lower scores are assigned to the negative class.

### 9.3 Evaluate at threshold 0.50

~~~python
print("Precision:", precision_score(y_true, predictions_050))
print("Recall:", recall_score(y_true, predictions_050))
print("F1:", f1_score(y_true, predictions_050))
~~~

**Expected output**
Precision, recall and F1 values calculated from the thresholded predictions.

**Explanation**
The metrics describe the selected operating threshold. They are not permanent properties of the model independent of threshold.

### 9.4 Lower the threshold to 0.30

~~~python
threshold = 0.30

predictions_030 = (
    fraud_scores >= threshold
).astype(int)

print(predictions_030)
~~~

**Expected output**
~~~text
[0 0 1 1 1 1 1 1 1 0]
~~~

**Explanation**
More observations are predicted as fraud. This may catch additional actual fraud cases but may also create more false alerts.

### 9.5 Compare metrics at multiple thresholds

~~~python
threshold_results = []

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    predictions = (fraud_scores >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df)
~~~

**Expected output**
A table containing one row per threshold and the corresponding precision, recall and F1 values.

**Explanation**
This table helps compare operating points. Select a threshold using business requirements and validation evidence, not by blindly choosing the largest number in one column.

## 10. Class imbalance and accuracy

A dataset is imbalanced when one class occurs much more often than another.

Example:
- 9,900 normal transactions.
- 100 fraud transactions.

A model predicting every transaction as normal gets 99% accuracy but detects no fraud.

### 10.1 Create an imbalanced label example

~~~python
y_imbalanced = [0] * 9900 + [1] * 100
all_normal_predictions = [0] * 10000

print("Total:", len(y_imbalanced))
print("Fraud cases:", sum(y_imbalanced))
~~~

**Expected output**
~~~text
Total: 10000
Fraud cases: 100
~~~

**Explanation**
Only one percent of observations belong to the positive fraud class.

### 10.2 Calculate accuracy for the naive model

~~~python
naive_accuracy = accuracy_score(
    y_imbalanced,
    all_normal_predictions
)

print("Accuracy:", naive_accuracy)
~~~

**Expected output**
~~~text
Accuracy: 0.99
~~~

**Explanation**
The model gets all normal cases correct, but this score hides the fact that it misses every fraud case.

### 10.3 Calculate positive-class recall

~~~python
naive_recall = recall_score(
    y_imbalanced,
    all_normal_predictions,
    zero_division=0
)

print("Fraud recall:", naive_recall)
~~~

**Expected output**
~~~text
Fraud recall: 0.0
~~~

**Explanation**
The model detects none of the actual fraud cases. This demonstrates why accuracy alone is not enough for an imbalanced problem.

## 11. Macro, micro and weighted averages

For multi-class classification, scikit-learn can summarize class metrics using different averaging approaches.

### Macro average
- Calculate the metric separately for each class.
- Give every class equal importance in the final average.
- Useful when performance on every class matters equally, including minority classes.

### Weighted average
- Calculate each class metric.
- Weight each class according to its support.
- Larger classes have more influence.
- Can look strong when the majority class performs well.

### Micro average
- Aggregate the relevant counts across classes before calculating the metric.
- Useful for an overall view across all class decisions.
- In standard single-label multi-class classification, micro precision, recall and F1 are equal to accuracy.

### 11.1 Create a multi-class example

~~~python
y_multi_actual = [
    "Billing", "Billing", "Billing",
    "Technical", "Technical", "Account",
    "Account", "Account"
]

y_multi_predicted = [
    "Billing", "Technical", "Billing",
    "Technical", "Billing", "Account",
    "Account", "Technical"
]

print("Multi-class sample ready")
~~~

**Expected output**
~~~text
Multi-class sample ready
~~~

### 11.2 Compare average strategies

~~~python
from sklearn.metrics import precision_score, recall_score, f1_score

for average_type in ["macro", "weighted", "micro"]:
    print(
        average_type,
        "Precision:",
        precision_score(
            y_multi_actual,
            y_multi_predicted,
            average=average_type,
            zero_division=0
        ),
        "Recall:",
        recall_score(
            y_multi_actual,
            y_multi_predicted,
            average=average_type,
            zero_division=0
        ),
        "F1:",
        f1_score(
            y_multi_actual,
            y_multi_predicted,
            average=average_type,
            zero_division=0
        )
    )
~~~

**Expected output**
Three lines of metric results, one for macro, weighted and micro averaging.

**Explanation**
The averaging strategy changes how each class contributes. State the averaging method whenever reporting a multi-class precision, recall or F1 score.

## 12. Classification report

### 12.1 Generate per-class metrics

~~~python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_multi_actual,
        y_multi_predicted,
        zero_division=0
    )
)
~~~

**Expected output**
A report containing precision, recall, F1-score and support for Billing, Technical and Account, plus average rows.

**Explanation**
Read each class row to find which ticket type is performing poorly. Do not rely only on the final average rows.

## 13. Which metric should you choose?

| Business situation | Metric to emphasize | Reason |
|---|---|---|
| Fraud screening where missed fraud is costly | Recall, with precision monitoring | Find as many fraud cases as practical |
| Manual review team has limited capacity | Precision | Avoid flooding reviewers with false alerts |
| Medical screening | Recall, with follow-up testing | Reduce missed potential cases |
| Spam filtering where important emails must not be blocked | Precision | Reduce legitimate emails flagged as spam |
| Balanced classification with similar error costs | Accuracy plus class metrics | Overall correctness may be informative |
| Minority class is important and both errors matter | F1 and class-specific metrics | Review the precision-recall balance |
| Multi-class task where every class matters | Macro average and per-class metrics | Prevent majority classes hiding weak minority performance |

The metric is a starting point for evaluation, not a substitute for a decision policy.

## 14. Common interview mistakes

- Saying accuracy is always the best metric.
- Confusing precision with recall.
- Forgetting to identify the positive class.
- Reporting F1 without explaining the averaging method.
- Ignoring class support.
- Treating a probability threshold of 0.50 as universally correct.
- Tuning the threshold using the final test set.
- Assuming increasing recall is free of cost.
- Choosing a metric without knowing the business impact of errors.

## 15. Interview-ready explanations

**What is accuracy?**
- The proportion of all predictions that are correct.
- Useful when class distribution and error costs make overall correctness meaningful.
- Can be misleading when one class dominates.

**What is precision?**
- Of the observations predicted positive, the proportion that are actually positive.
- Important when false positive alerts or actions are costly.

**What is recall?**
- Of all actual positive observations, the proportion detected by the model.
- Important when missing a positive case is costly.

**What is F1-score?**
- A combined measure of precision and recall.
- Useful when both matter, but it does not directly describe true-negative performance.

**What is the difference between macro and weighted F1?**
- Macro gives each class equal weight.
- Weighted accounts for class support.
- Weighted F1 can hide poor minority-class performance.

## 16. Scenario-based interview practice

### Scenario 1: A fraud model has 99% accuracy but recall is zero
**Expected response:**
- The model may be predicting every transaction as normal.
- Accuracy is dominated by the majority class.
- Inspect the confusion matrix and fraud support.
- Focus on positive-class recall and precision.
- Revisit features, threshold and training strategy using validation data.

### Scenario 2: A review team can investigate only a small number of alerts
**Expected response:**
- Precision and alert volume are important.
- Review the number of false positives.
- Rank alerts by model score if that is validated for the use case.
- Select an operating threshold based on review capacity and the cost of missed cases.
- Monitor recall so that the reduction in alerts does not hide excessive missed positives.

### Scenario 3: A medical screening model misses actual disease cases
**Expected response:**
- Treat false negatives as a serious risk.
- Review recall for the positive disease class.
- Investigate data coverage and threshold selection.
- Involve clinical experts in the operating policy.
- Do not deploy based only on an aggregate F1 or accuracy score.

### Scenario 4: Weighted F1 is high but minority-class F1 is poor
**Expected response:**
- The majority class may be dominating the weighted average.
- Inspect per-class support, precision, recall and F1.
- Report macro and weighted results together.
- Investigate whether minority-class examples are insufficient or poorly represented.

## 17. Lesson completion checklist

- [ ] I can explain accuracy, precision, recall and F1-score.
- [ ] I can identify the business meaning of false positives and false negatives.
- [ ] I can calculate metrics using scikit-learn.
- [ ] I can explain threshold changes.
- [ ] I understand why accuracy can mislead on imbalanced data.
- [ ] I can distinguish macro, micro and weighted averages.
- [ ] I can select metrics based on business requirements.
- [ ] I can answer scenario questions using evidence and trade-offs.

**Next lesson:** Day 46 — ROC Curve, AUC and PR Curve.
