# Day 46: Interview Questions — ROC Curve, AUC and PR Curve

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is ROC curve?
**Answer:**
- ROC means Receiver Operating Characteristic.
- It plots True Positive Rate against False Positive Rate.
- Each point represents a different classification threshold.
- It shows the trade-off between detecting positive cases and generating false positive alerts.

**Follow-up:** What does a curve close to the top-left generally indicate?  
**Answer:** Better separation between positive and negative classes across thresholds.

### Q2. What is ROC-AUC?
**Answer:**
- AUC means Area Under the Curve.
- ROC-AUC summarizes the ranking ability of a classifier across thresholds.
- It can be interpreted as how often a randomly selected positive observation receives a higher score than a randomly selected negative observation.
- It is not accuracy and does not identify the best deployment threshold.

**Follow-up:** Does an AUC of 0.90 mean 90% accuracy?  
**Answer:** No. It indicates strong ranking separation under the ROC-AUC interpretation, not 90% classification accuracy.

### Q3. What is a Precision-Recall curve?
**Answer:**
- It plots precision against recall across different thresholds.
- It focuses on positive-class prediction quality.
- It helps understand how many predicted positives are correct as the model attempts to capture more actual positives.
- It is especially useful when the positive class is rare.

**Follow-up:** Why is it useful for fraud detection?  
**Answer:** Fraud may be a small fraction of transactions. The PR curve makes the effect of false alerts on precision visible while recall changes.

### Q4. What is the difference between ROC curve and PR curve?
**Answer:**
- ROC uses TPR and FPR.
- PR uses precision and recall.
- ROC summarizes positive detection relative to the false-positive rate.
- PR focuses on the reliability and coverage of positive predictions.
- For rare positive classes, PR analysis is often more informative about alert quality.

### Q5. What is Average Precision?
**Answer:**
- Average Precision summarizes the precision-recall curve across recall increments.
- It is a common summary metric for imbalanced classification.
- It is not identical to every numerical integration method called PR-AUC.
- Report the exact metric used.

**Follow-up:** Why might AP and trapezoidal PR area differ?  
**Answer:** They use different ways to summarize the precision-recall points, so the numerical results can differ.

### Q6. Does ROC-AUC choose the classification threshold?
**Answer:**
- No. ROC-AUC summarizes ranking performance across thresholds.
- A deployed classifier still needs an operating threshold.
- Choose that threshold using validation data, business costs and operational constraints.
- Evaluate the locked decision on the final test set.

## 2. Practical and technical questions

### Q7. Why should ROC and PR curves use probabilities or continuous scores rather than only predicted labels?
**Answer:**
- Hard labels represent only one threshold.
- Continuous scores allow the curve to examine many thresholds.
- This reveals the trade-off across possible operating points.
- Use the correct score direction for the positive class.

### Q8. What does a ROC curve's diagonal reference line represent?
**Answer:**
- It represents random ranking in the usual ROC plot.
- A curve above it generally indicates better ranking than random.
- A curve below it may indicate reversed score direction or poor ranking.
- Verify label and score orientation before drawing conclusions.

### Q9. Why can ROC-AUC look strong on highly imbalanced data?
**Answer:**
- The negative class is very large.
- A small false-positive rate can still create a large absolute number of false alerts.
- ROC displays the false-positive rate, not the total alert workload.
- PR curve, precision and threshold-specific counts make positive alert quality clearer.

### Q10. Is ROC-AUC a probability calibration metric?
**Answer:**
- No. It evaluates ranking.
- A model can rank observations well but output poorly calibrated probabilities.
- If decisions depend on probability meaning, evaluate calibration separately.

### Q11. What is the baseline of a Precision-Recall curve?
**Answer:**
- The baseline precision for a no-skill ranking is related to the positive-class prevalence in the evaluated data.
- If positives are rare, baseline precision is low.
- Always report the class prevalence when interpreting Average Precision.

### Q12. Two models have similar ROC-AUC. How do you choose?
**Answer:**
- Compare PR curves and Average Precision when positives are rare.
- Compare precision and recall at candidate operating thresholds.
- Review false-positive and false-negative counts.
- Check calibration if probability values drive decisions.
- Consider stability, latency, interpretability and business costs.

## 3. Threshold selection and troubleshooting

### Q13. A model has high ROC-AUC but too many false alerts in production. What do you investigate?
**Answer:**
- Inspect the production threshold and confusion matrix.
- Review precision and alert volume.
- Compare PR curve and Average Precision.
- Check whether production class prevalence differs from evaluation data.
- Select an operating threshold using representative validation data and review capacity.

### Q14. Why should you not select the threshold using the final test set?
**Answer:**
- Threshold selection is a model decision.
- Repeatedly tuning on the test set leaks information from the final evaluation.
- Select the threshold on validation data.
- Use the untouched test set for final performance reporting.

### Q15. AUC is below 0.5. What do you check?
**Answer:**
- Confirm which label is positive.
- Check whether higher scores actually indicate the positive class.
- Verify that labels and scores are aligned to the same observations.
- Inspect data quality and model performance.
- Do not automatically reverse scores without validating the reason.

### Q16. The PR curve is poor even though ROC-AUC is high. How do you explain it?
**Answer:**
- ROC-AUC may look favourable because the negative class is large.
- The model may still generate many false positive alerts relative to the number of positives.
- Inspect precision, recall and false alert counts.
- Use PR analysis to understand positive-class usefulness.

## 4. Scenario-based interview questions

### Scenario 1: ROC-AUC is high but the business sees too many false alerts
**Expected answer:**
- ROC-AUC summarizes ranking across thresholds, not the chosen operating point.
- Inspect the confusion matrix, precision and alert count at the deployed threshold.
- Evaluate the PR curve and Average Precision if positives are rare.
- Select a threshold using validation data and review capacity.

### Scenario 2: Fraud is rare and ROC-AUC looks strong
**Expected answer:**
- Check positive-class prevalence.
- Calculate the PR curve and Average Precision.
- Inspect precision, recall and false alert counts at candidate thresholds.
- Explain the actual number of false positives, not only the false-positive rate.

### Scenario 3: Two models have similar ROC-AUC
**Expected answer:**
- Compare PR behaviour if the positive class is rare.
- Compare threshold-specific precision, recall and confusion matrices.
- Check calibration if probability values are used operationally.
- Compare stability and business cost before choosing.

### Scenario 4: The team chooses the threshold that maximizes F1 on the test set
**Expected answer:**
- This uses the final test set for model selection.
- Choose the threshold on validation data.
- Lock the decision and evaluate once on the untouched test set.
- Document the business objective and accepted trade-off.

### Scenario 5: A manager interprets ROC-AUC 0.90 as 90% of individual predictions being correct
**Expected answer:**
- Clarify that ROC-AUC is a ranking metric, not accuracy.
- Explain its positive-versus-negative pairwise interpretation.
- Show accuracy, confusion matrix and threshold-specific precision and recall separately.
- Present the metric in business language to prevent an incorrect conclusion.

## 5. Strong points to remember in interviews

- ROC curve plots TPR against FPR.
- ROC-AUC summarizes ranking across thresholds.
- PR curve plots precision against recall.
- PR analysis is especially important when positives are rare.
- AUC is not accuracy and is not probability calibration.
- A high AUC does not guarantee a useful operating threshold.
- A small false-positive rate can still mean many false alerts in a large negative population.
- Average Precision and trapezoidal PR area may differ.
- Select thresholds using validation data, not the final test set.
- Always report class prevalence and threshold-specific business impact.

**Senior developer principle:** AUC tells us how well a model ranks cases; the threshold and business policy determine what the system actually does.
