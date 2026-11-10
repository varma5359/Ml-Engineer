# Day 44: Confusion Matrix and Classification Metrics

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain why classification models must be evaluated beyond accuracy.
- Understand actual and predicted class labels.
- Read a confusion matrix correctly.
- Explain true positive, true negative, false positive and false negative.
- Generate binary and multi-class confusion matrices using scikit-learn.
- Interpret classification errors in business scenarios.
- Select evaluation priorities based on the cost of mistakes.

## 2. Why do we evaluate a classification model?

A classification model predicts a category such as Fraud / Not fraud, Spam / Not spam, Customer churn / No churn, Loan default / No default, or Defective / Good product.

Predictions alone do not tell us whether the model is useful. A fraud model may correctly classify most normal transactions while missing many fraudulent ones. Its overall accuracy may still look high because fraud is rare.

Evaluation helps answer:
- How many predictions were correct?
- Which types of mistakes did the model make?
- Which mistakes are more expensive?
- Is the model suitable for the business process?
- What should be improved before deployment?

## 3. Actual class versus predicted class

- **Actual class:** The known label in the evaluation dataset.
- **Predicted class:** The category returned by the model.
- **Correct prediction:** Actual and predicted classes match.
- **Incorrect prediction:** Actual and predicted classes differ.

| Transaction | Actual | Predicted |
|---|---|---|
| T101 | Fraud | Fraud |
| T102 | Normal | Normal |
| T103 | Fraud | Normal |
| T104 | Normal | Fraud |

T103 is a missed fraud case. T104 is a normal transaction incorrectly flagged as fraud.

## 4. What is a confusion matrix?

A **confusion matrix** is a table that compares actual class labels with predicted class labels.

| Actual / Predicted | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

Remember:
- **True:** The prediction is correct.
- **False:** The prediction is incorrect.
- **Positive:** The model predicted the positive class.
- **Negative:** The model predicted the negative class.

Positive and negative describe class labels, not whether an outcome is good or bad.

## 5. Understand the four outcomes with a fraud example

Assume the positive class is Fraud.

### 5.1 True Positive (TP)
- Actual transaction is fraud.
- Model predicts fraud.
- Fraud is correctly detected.
- Example: A fraudulent payment is flagged for investigation.

### 5.2 True Negative (TN)
- Actual transaction is normal.
- Model predicts normal.
- A legitimate payment is correctly allowed.

### 5.3 False Positive (FP)
- Actual transaction is normal.
- Model predicts fraud.
- A legitimate payment is incorrectly flagged.
- Business impact: customer inconvenience, manual review cost or declined valid payments.

### 5.4 False Negative (FN)
- Actual transaction is fraud.
- Model predicts normal.
- Fraud is missed.
- Business impact: financial loss, compliance exposure or customer harm.

## 6. Why false positives and false negatives matter

| Application | False positive impact | False negative impact |
|---|---|---|
| Fraud detection | Valid payment investigated or blocked | Fraudulent payment passes |
| Medical screening | Healthy person sent for further testing | Disease case is missed |
| Spam filtering | Important email marked as spam | Spam reaches inbox |
| Customer churn | Stable customer receives retention offer | At-risk customer is missed |
| Defect detection | Good product rejected | Defective product reaches customer |

There is no universal rule that one error is always worse. Understand the business context first.

## 7. Create a small classification example

### 7.1 Import required libraries

~~~python
import numpy as np
import pandas as pd

print("Libraries ready")
~~~

**Expected output**
~~~text
Libraries ready
~~~

**Explanation**
- NumPy supports numerical arrays.
- Pandas helps display evaluation results.
- We will use scikit-learn for the confusion matrix.

### 7.2 Create actual labels

~~~python
y_actual = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]

print(y_actual)
~~~

**Expected output**
~~~text
[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
~~~

**Explanation**
- 1 represents Fraud, the positive class.
- 0 represents Normal, the negative class.
- These are the known labels used for evaluation.

### 7.3 Create model predictions

~~~python
y_predicted = [1, 0, 0, 0, 1, 1, 0, 1, 0, 0]

print(y_predicted)
~~~

**Expected output**
~~~text
[1, 0, 0, 0, 1, 1, 0, 1, 0, 0]
~~~

**Explanation**
- These are the model's predicted labels.
- The third observation is a missed positive case.
- The sixth observation is an incorrectly flagged negative case.

## 8. Generate the confusion matrix

### 8.1 Import confusion_matrix

~~~python
from sklearn.metrics import confusion_matrix

print("Confusion matrix function imported")
~~~

**Expected output**
~~~text
Confusion matrix function imported
~~~

### 8.2 Calculate the matrix

~~~python
cm = confusion_matrix(y_actual, y_predicted)

print(cm)
~~~

**Expected output**
~~~text
[[5 1]
 [1 3]]
~~~

**Explanation**

By default, scikit-learn arranges binary labels in sorted order:
- First row: actual class 0.
- Second row: actual class 1.
- First column: predicted class 0.
- Second column: predicted class 1.

Therefore:
- 5 = True Negative.
- 1 = False Positive.
- 1 = False Negative.
- 3 = True Positive.

Always confirm the class order before explaining a confusion matrix.

### 8.3 Display the matrix with labels

~~~python
cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Fraud"],
    columns=["Predicted Normal", "Predicted Fraud"]
)

print(cm_df)
~~~

**Expected output**
A labelled 2-by-2 table showing 5, 1, 1 and 3 in the positions described above.

**Explanation**
A labelled table reduces the risk of confusing false positives with false negatives.

### 8.4 Visualize the confusion matrix

~~~python
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Fraud Detection Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.show()
~~~

**Expected output**
A heatmap containing four cells with counts 5, 1, 1 and 3.

**Explanation**
- annot=True writes values inside the cells.
- fmt="d" displays integer counts.
- The axes show predicted and actual classes.
- Visualization makes the error pattern easier to discuss.

## 9. Extract the four confusion matrix values

### 9.1 Unpack TN, FP, FN and TP

~~~python
tn, fp, fn, tp = cm.ravel()

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)
~~~

**Expected output**
~~~text
TN: 5
FP: 1
FN: 1
TP: 3
~~~

**Explanation**
- ravel() flattens the matrix into a one-dimensional sequence.
- This order assumes the standard binary label ordering [0, 1].
- If labels or matrix layout change, inspect the matrix before unpacking.

### 9.2 Count total observations

~~~python
total = tn + fp + fn + tp

print("Total observations:", total)
~~~

**Expected output**
~~~text
Total observations: 10
~~~

**Explanation**
Every evaluated observation belongs to one of the four outcomes.

### 9.3 Count correct predictions

~~~python
correct_predictions = tn + tp

print("Correct predictions:", correct_predictions)
~~~

**Expected output**
~~~text
Correct predictions: 8
~~~

**Explanation**
True negatives and true positives are the correct predictions.

### 9.4 Count incorrect predictions

~~~python
incorrect_predictions = fp + fn

print("Incorrect predictions:", incorrect_predictions)
~~~

**Expected output**
~~~text
Incorrect predictions: 2
~~~

**Explanation**
False positives and false negatives are the mistakes. Day 45 explains how metrics describe these outcomes in more detail.

## 10. Use scikit-learn classification reports

### 10.1 Import classification_report

~~~python
from sklearn.metrics import classification_report

print("Classification report imported")
~~~

**Expected output**
~~~text
Classification report imported
~~~

### 10.2 Generate the report

~~~python
print(
    classification_report(
        y_actual,
        y_predicted,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)
~~~

**Expected output**
A text report with rows for Normal and Fraud and columns for precision, recall, F1-score and support.

**Explanation**
- target_names gives readable class names in the same order as the labels.
- support is the number of actual observations for each class.
- zero_division=0 defines what to display when a metric denominator is zero.
- Read class-specific rows rather than relying only on overall accuracy.

## 11. Multi-class confusion matrices

A confusion matrix also works when there are more than two classes.

Example: A model classifies support tickets as Billing, Technical or Account.

### 11.1 Create multi-class labels

~~~python
actual_categories = [
    "Billing", "Technical", "Account",
    "Billing", "Technical", "Account",
    "Billing", "Account"
]

predicted_categories = [
    "Billing", "Technical", "Billing",
    "Account", "Technical", "Account",
    "Billing", "Account"
]

print("Actual:", actual_categories)
print("Predicted:", predicted_categories)
~~~

**Expected output**
Two lists containing eight ticket categories each.

**Explanation**
The labels are categories rather than numeric 0 and 1. The same evaluation principle applies.

### 11.2 Calculate the multi-class matrix

~~~python
multi_cm = confusion_matrix(
    actual_categories,
    predicted_categories,
    labels=["Billing", "Technical", "Account"]
)

print(multi_cm)
~~~

**Expected output**
A 3-by-3 matrix. Rows represent actual categories and columns represent predicted categories in the supplied order.

**Explanation**
- Diagonal cells are correct predictions.
- Off-diagonal cells show which categories are confused.
- A Billing ticket predicted as Account appears in the Billing row and Account column.

### 11.3 Display multi-class matrix labels

~~~python
multi_cm_df = pd.DataFrame(
    multi_cm,
    index=["Actual Billing", "Actual Technical", "Actual Account"],
    columns=["Predicted Billing", "Predicted Technical", "Predicted Account"]
)

print(multi_cm_df)
~~~

**Expected output**
A labelled 3-by-3 table.

**Explanation**
The matrix can reveal that a model performs well for two ticket types but frequently confuses the third.

## 12. Read a confusion matrix as a business report

When explaining results to a business stakeholder, do not only report four numbers.

Include:
- Number of observations evaluated.
- Which class is treated as positive.
- Number of correct positive detections.
- Number of missed positive cases.
- Number of false alerts.
- Which error is most costly.
- Whether evaluation data represents expected production conditions.
- What action is recommended.

Example:
"The model correctly flagged 3 of the 4 fraud cases in this small sample. It missed one fraud case and incorrectly flagged one normal transaction. We need a larger representative evaluation before deciding whether this operating point is acceptable."

## 13. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Read rows as predicted and columns as actual | Check the function convention and axis labels |
| Forget which class is positive | State the positive class explicitly |
| Assume false positive is always worse | Compare business costs |
| Report only accuracy | Inspect class-specific errors |
| Ignore class imbalance | Check the distribution of actual labels |
| Treat confusion matrix as a probability table | It contains counts unless normalized |
| Compare matrices with different label orders | Specify and document label order |
| Evaluate on training predictions only | Use held-out data or a suitable validation strategy |

## 14. Interview-ready explanation

**What is a confusion matrix?**  
It is a table that compares actual labels with predicted labels and shows the types and counts of correct and incorrect classification outcomes.

**Why is it important?**  
It helps identify whether a model is missing positive cases, generating too many false alerts or confusing particular classes.

**What is the difference between false positive and false negative?**
- False positive: model predicts positive, but actual class is negative.
- False negative: model predicts negative, but actual class is positive.

**Can a confusion matrix be used for multi-class classification?**  
Yes. Each row represents an actual class and each column represents a predicted class, according to the chosen label order.

## 15. Scenario-based interview practice

### Scenario 1: Fraud is missed
A payment model predicts a fraudulent transaction as normal.
- This is a false negative.
- Review the missed case and the features available at prediction time.
- Check whether the threshold is suitable for the cost of missed fraud.
- Do not assume one example proves the model is generally poor.

### Scenario 2: Too many valid customers are blocked
A fraud model flags many legitimate payments.
- These are false positives.
- Review customer impact and manual investigation workload.
- Check whether the decision threshold or feature behaviour needs adjustment.
- Consider a review queue rather than automatically declining every flagged payment.

### Scenario 3: High accuracy but poor fraud detection
A model has high accuracy, but the business reports missed fraud.
- Inspect the confusion matrix and class distribution.
- Check recall for the fraud class.
- Accuracy may be dominated by the large normal class.
- Select metrics based on business risk.

### Scenario 4: Multi-class ticket routing
The model confuses Billing and Account tickets.
- Inspect the off-diagonal cells.
- Review examples from both classes.
- Check label definitions, overlapping language and data quality.
- Improve training examples or routing rules where justified.

## 16. Lesson completion checklist

- [ ] I can explain actual and predicted labels.
- [ ] I can define TP, TN, FP and FN.
- [ ] I can create and label a confusion matrix.
- [ ] I can interpret a multi-class confusion matrix.
- [ ] I can explain why the positive class must be clear.
- [ ] I can describe the business impact of classification errors.
- [ ] I can explain why accuracy alone may be misleading.
- [ ] I can communicate model errors in plain language.

**Next lesson:** Day 45 — Accuracy, Precision, Recall and F1-Score.
