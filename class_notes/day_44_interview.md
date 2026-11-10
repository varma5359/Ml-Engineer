# Day 44: Interview Questions — Confusion Matrix and Classification Metrics

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is a confusion matrix?
**Answer:**
- A table that compares actual class labels with model-predicted labels.
- It summarizes correct predictions and classification errors.
- In binary classification it contains TP, TN, FP and FN.
- It helps us understand what kind of mistakes the model makes, not only how many predictions are correct.

**Example:** In fraud detection, it shows how many fraudulent transactions were detected, missed and how many normal transactions were incorrectly flagged.

**Follow-up:** Can it be used for multi-class classification?  
**Answer:** Yes. Each class gets a row and column. Diagonal cells are correct predictions and off-diagonal cells show class confusion.

### Q2. Explain True Positive, True Negative, False Positive and False Negative.
**Answer:**
- **True Positive:** Actual positive and predicted positive. Example: a real fraud transaction is flagged.
- **True Negative:** Actual negative and predicted negative. Example: a legitimate transaction is allowed.
- **False Positive:** Actual negative but predicted positive. Example: a legitimate payment is incorrectly flagged.
- **False Negative:** Actual positive but predicted negative. Example: a fraudulent payment is missed.

**Interview tip:** Always state what the positive class represents in the problem.

**Follow-up:** Which error is more serious?  
**Answer:** It depends on the business. In fraud detection, a false negative may cause financial loss, while a false positive may inconvenience a genuine customer. Other applications have different costs.

### Q3. What is the difference between false positive and false negative?
**Answer:**
- False positive: the model raises an alert when the actual class is negative.
- False negative: the model misses an actual positive case.
- False positives can create unnecessary reviews, blocked transactions or customer friction.
- False negatives can allow fraud, disease or defects to go undetected.
- The business should define the relative cost of these mistakes.

**Follow-up:** How can you reduce false negatives?  
**Answer:** Investigate the decision threshold, improve relevant features and training data, and evaluate recall. Lowering the threshold may catch more positives but can increase false positives.

### Q4. Why is a confusion matrix useful when accuracy is available?
**Answer:**
- Accuracy provides one overall summary.
- The confusion matrix shows the distribution of correct and incorrect predictions.
- It reveals which class is being missed.
- It shows whether the model creates too many false alerts.
- It supports decisions based on the actual cost of mistakes.

**Follow-up:** Can a confusion matrix alone prove that a model is ready for production?  
**Answer:** No. We also need representative validation data, suitable metrics, threshold analysis, operational checks and monitoring.

### Q5. How do you interpret a binary confusion matrix in scikit-learn?
**Answer:**
- By default, rows represent actual labels and columns represent predicted labels.
- For labels 0 and 1, the matrix is commonly arranged as TN, FP in the first row and FN, TP in the second row.
- Confirm the label order before unpacking values.
- Label the axes to avoid interpretation mistakes.

**Follow-up:** What does ravel() do?  
**Answer:** It flattens the matrix into a one-dimensional sequence. For the standard binary order, it can be unpacked as tn, fp, fn, tp.

### Q6. What is support in a classification report?
**Answer:**
- Support is the number of actual observations belonging to a class in the evaluation data.
- It helps explain how much evidence contributes to each class metric.
- A metric based on very few positive examples may be unstable.
- Always consider support when presenting class-specific results.

**Follow-up:** If a class has zero actual examples in the test set, what is the issue?  
**Answer:** We cannot evaluate performance for that class from this test set. Review the split and obtain a representative evaluation sample.

## 2. Practical and troubleshooting questions

### Q7. Your model has 98% accuracy, but the fraud team says many fraud cases are missed. How do you investigate?
**Answer:**
- Inspect the actual class distribution.
- Generate the confusion matrix.
- Review false negatives for the fraud class.
- Calculate recall and precision for fraud.
- Check whether the evaluation data reflects the production fraud rate.
- Review threshold selection and the cost of missed fraud.
- Do not assume high accuracy means good fraud detection.

### Q8. A fraud model blocks too many genuine transactions. What does the confusion matrix show?
**Answer:**
- A high false-positive count.
- Genuine transactions predicted as fraud.
- Investigate the threshold, feature quality and customer segments affected.
- Consider a manual review workflow for uncertain cases.
- Compare the reduction in false positives with any increase in missed fraud.

### Q9. A multi-class support ticket classifier confuses Billing and Account categories. What would you do?
**Answer:**
- Inspect the relevant off-diagonal cells.
- Review misclassified examples.
- Check whether category definitions overlap.
- Examine label quality and the language used in tickets.
- Improve training examples or feature representation.
- Re-evaluate each class separately after changes.

### Q10. The confusion matrix changes when you reorder class labels. Is the model different?
**Answer:**
- Not necessarily.
- The displayed matrix row and column order depends on the labels supplied.
- The predictions can remain exactly the same.
- Keep a documented class order and label axes clearly.

### Q11. Can you compare confusion matrix counts from two test sets directly?
**Answer:**
- Only with caution.
- Test sets may have different sizes and class distributions.
- Compare normalized rates and class-specific metrics as well as counts.
- Confirm both test sets represent comparable populations.

### Q12. Why should you not evaluate only on training predictions?
**Answer:**
- The model has already learned from training data.
- Training performance can be overly optimistic.
- Use a held-out test set or a suitable validation design to estimate generalization.
- Keep the final test set separate from model and threshold selection.

## 3. Interview scenarios — explain your decision

### Scenario 1: Missed fraud
A transaction is actually fraudulent, but the model predicts normal.

**Expected interview response:**
- Identify it as a false negative.
- Review whether the relevant signals were present in the input.
- Check the threshold and fraud-class recall.
- Investigate whether similar cases are missed systematically.
- Recommend changes only after evaluation on representative data.

### Scenario 2: Legitimate customer blocked
A valid customer payment is flagged as fraud.

**Expected interview response:**
- Identify it as a false positive.
- Explain customer friction and investigation cost.
- Review false-positive patterns by customer segment and transaction type.
- Evaluate whether a different threshold or review queue is appropriate.
- Monitor the trade-off with false negatives.

### Scenario 3: A model performs poorly on one class
Overall performance looks acceptable, but one ticket category has many errors.

**Expected interview response:**
- Inspect the multi-class confusion matrix.
- Review per-class support and metrics.
- Examine label quality and class overlap.
- Identify whether the issue is data coverage, feature representation or model behaviour.
- Report the weak class clearly rather than hiding it in an overall score.

## 4. Strong points to remember in interviews

- Always define the positive class before discussing TP, FP, FN or recall.
- A confusion matrix is a count table, not a probability table unless normalized.
- The diagonal contains correct predictions; off-diagonal cells contain errors.
- False positives and false negatives have different business costs.
- Accuracy does not explain which class is failing.
- For multi-class tasks, inspect each class and the off-diagonal confusion patterns.
- Use held-out evaluation data, not only training predictions.
- A model should not be approved based on one metric or one confusion matrix alone.

**Senior developer principle:** Do not say only "the model achieved 95% accuracy." Explain which cases it gets right, which cases it misses, and what those mistakes mean to the business.
