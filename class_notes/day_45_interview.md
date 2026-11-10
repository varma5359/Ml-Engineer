# Day 45: Interview Questions — Accuracy, Precision, Recall and F1-Score

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is accuracy?
**Answer:**
- Accuracy is the proportion of all predictions that are correct.
- It includes correct positive and correct negative predictions.
- It is useful when classes and error costs are reasonably balanced.
- It can be misleading when one class is much more common.

**Example:** A model that predicts every transaction as normal can achieve very high accuracy when fraud is rare, even though it detects no fraud.

**Follow-up:** When would you avoid using accuracy alone?  
**Answer:** In fraud, disease detection, defect detection or any imbalanced problem where the minority class is important.

### Q2. What is precision?
**Answer:**
- Precision measures how reliable positive predictions are.
- It answers: among all cases predicted positive, how many are actually positive?
- High precision means fewer false positive alerts.
- It is important when a positive prediction triggers expensive review, customer disruption or a costly action.

**Example:** If a fraud system flags 100 transactions and 90 are actually fraudulent, its precision is high.

**Follow-up:** How can a business improve precision?  
**Answer:** Review the threshold, improve features and training data, and investigate false positives. Any change must also be checked for its impact on recall.

### Q3. What is recall?
**Answer:**
- Recall measures how many actual positive cases the model finds.
- It answers: among all actual positive cases, how many were detected?
- High recall means fewer false negatives.
- It is important when missing a positive case has a high cost.

**Example:** In medical screening, recall helps measure how many actual disease cases are identified for further testing.

**Follow-up:** What is the risk of focusing only on recall?  
**Answer:** The model may flag too many negative cases, creating false positives and high review workload.

### Q4. What is F1-score?
**Answer:**
- F1-score combines precision and recall.
- It is useful when both false positives and false negatives matter.
- It gives a single summary of the precision-recall balance.
- It does not directly include true negatives, so it should not replace all other evaluation views.

**Follow-up:** Is a high F1-score always sufficient for deployment?  
**Answer:** No. Check per-class results, support, threshold behaviour, business costs, data representativeness and operational constraints.

### Q5. Explain precision versus recall with a business example.
**Answer:**
- Precision focuses on the correctness of positive alerts.
- Recall focuses on how many actual positive cases are found.
- In fraud detection, high precision means most flagged transactions are genuinely fraudulent.
- High recall means most fraudulent transactions are caught.
- The correct balance depends on financial loss, customer impact and review capacity.

**Follow-up:** Can precision and recall move in opposite directions?  
**Answer:** Yes. Lowering the decision threshold often increases recall but may lower precision because more cases are flagged positive.

### Q6. Why can accuracy be misleading on imbalanced data?
**Answer:**
- The majority class contributes most observations.
- A model can predict the majority class for every record and still achieve high accuracy.
- It may completely fail to detect the minority class.
- Inspect the confusion matrix, minority-class recall, precision and F1.

**Follow-up:** What else would you report?  
**Answer:** Class support, per-class metrics, precision-recall curve and business-relevant threshold performance.

## 2. Metric selection questions

### Q7. When would you prioritize precision?
**Answer:**
- When false positive alerts are expensive.
- When a human team has limited review capacity.
- When blocking a legitimate customer causes significant disruption.
- Examples include high-cost fraud investigations and spam filtering where legitimate emails must not be blocked.

### Q8. When would you prioritize recall?
**Answer:**
- When missing a positive case is especially costly.
- Examples include initial medical screening, fraud detection and safety defect identification.
- High recall should be balanced with the capacity to handle false positives.

### Q9. When would you use F1-score?
**Answer:**
- When both precision and recall matter.
- When a single summary of their balance is useful.
- It is not sufficient when true-negative performance or explicit business costs are central to the decision.

### Q10. What is the difference between macro and weighted average?
**Answer:**
- Macro calculates the metric for each class and gives each class equal weight.
- Weighted average accounts for the number of actual observations in each class.
- Weighted results can be dominated by the majority class.
- Report per-class metrics when minority-class performance matters.

**Follow-up:** What is micro averaging?  
**Answer:** It aggregates the counts across classes before calculating the metric. For standard single-label multi-class classification, micro precision, recall and F1 equal accuracy.

## 3. Threshold and troubleshooting questions

### Q11. What happens when you lower a classification threshold?
**Answer:**
- More observations are likely to be predicted positive.
- Recall often increases because more actual positive cases are captured.
- False positives may increase, reducing precision.
- Evaluate the resulting trade-off on validation data.

### Q12. Your model has high recall but very low precision. What does this mean?
**Answer:**
- It finds many actual positive cases.
- It also flags many negative cases incorrectly.
- Review false positives, threshold, feature quality and review capacity.
- Decide whether the additional recall is worth the extra alerts.

### Q13. Your model has high precision but low recall. What does this mean?
**Answer:**
- Most positive predictions are correct.
- Many actual positive cases are still being missed.
- Review false negatives and threshold selection.
- Determine whether missed cases are more costly than extra false alerts.

### Q14. Why should the threshold be tuned on validation data rather than the test set?
**Answer:**
- Choosing a threshold is part of model selection.
- Repeatedly using the test set to choose it leaks information from the final evaluation.
- Select the threshold on validation data, then evaluate once on the untouched test set.

### Q15. A class has very few examples. How do you interpret its F1-score?
**Answer:**
- Check its support.
- A small number of examples can make the score unstable.
- Inspect individual errors and use suitable validation strategies.
- Avoid presenting a single score without its sample count and limitations.

## 4. Scenario-based interview questions

### Scenario 1: Fraud model has 99% accuracy but recall is zero
**Expected answer:**
- The model may predict every transaction as normal.
- Accuracy is dominated by the majority class.
- Inspect the confusion matrix and class support.
- Focus on fraud-class recall and precision.
- Revisit features, threshold and training strategy using validation data.

### Scenario 2: A review team can investigate only a small number of alerts
**Expected answer:**
- Precision and alert volume are important.
- Review false positive counts and alert ranking quality.
- Select an operating threshold based on review capacity and missed-case cost.
- Monitor recall so fewer alerts do not mean too many missed positives.

### Scenario 3: Medical screening misses actual disease cases
**Expected answer:**
- Treat false negatives as a serious risk.
- Review recall for the positive disease class.
- Investigate data coverage and threshold selection.
- Involve clinical experts in the operating policy.
- Do not approve the model based only on aggregate accuracy or F1.

### Scenario 4: Weighted F1 is high but minority-class F1 is poor
**Expected answer:**
- The majority class may dominate the weighted average.
- Inspect per-class support, precision, recall and F1.
- Report macro and weighted results together.
- Investigate whether minority-class examples are insufficient or poorly represented.

### Scenario 5: A product team asks for one metric to compare every model
**Expected answer:**
- Ask what the model is used for and what each error costs.
- Identify the positive class and class distribution.
- Choose a primary metric aligned with the business goal.
- Report supporting metrics and the confusion matrix.
- Do not claim that one metric is universally best.

## 5. Strong points to remember in interviews

- Accuracy = overall correctness; it can hide minority-class failure.
- Precision = reliability of positive predictions; it is affected by false positives.
- Recall = coverage of actual positive cases; it is affected by false negatives.
- F1 = combined precision-recall summary; it does not directly include true negatives.
- Always state which class is positive.
- Always report support and class-specific results for imbalanced problems.
- A threshold changes precision and recall.
- Choose a threshold using validation data, not the final test set.
- Select metrics based on business costs, not habit.

**Senior developer principle:** Explain the metric in business terms, identify the error it emphasizes, and state what trade-off the team is accepting.
