# Day 82: Interview Questions — Capstone Evaluation and Model Selection

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. Why is a confusion matrix useful?
**Answer:**
- It shows true positives, false positives, true negatives and false negatives.
- It reveals the types of classification errors.
- It supports precision and recall interpretation.
- It connects model behavior to business costs.

**Follow-up:** Which cell represents a missed churner?

### Q2. Compare ROC-AUC and PR-AUC.
**Answer:**
- ROC-AUC measures ranking through true-positive and false-positive rates.
- PR-AUC measures precision-recall behavior.
- PR-AUC is often useful when positive cases are rare.
- Both summarize behavior across thresholds and do not replace threshold-specific analysis.

**Follow-up:** What is the no-skill baseline for a precision-recall curve related to?

### Q3. Why should threshold selection use validation data?
**Answer:**
- The threshold is a model decision selected during development.
- Repeated test-set selection leaks information from the final evaluation.
- Validation data supports threshold trade-offs and capacity checks.
- The final test set remains reserved for one final assessment.

**Follow-up:** What should be versioned with the threshold?

### Q4. What is segment evaluation?
**Answer:**
- It examines model performance for meaningful subgroups.
- It can reveal weaknesses hidden by aggregate metrics.
- It should include group sizes and uncertainty.
- Segments must be chosen responsibly and in line with governance.

**Follow-up:** What do you do when a segment has very few examples?

### Q5. What should a model selection report include?
**Answer:**
- Candidate comparison and validation strategy.
- Selected model and selection rationale.
- Final test metrics and confusion matrix.
- Threshold and business implications.
- Segment performance, limitations and deployment recommendation.

**Follow-up:** Why separate validation and final test results in the report?

## 2. Scenario questions

### Q6. ROC-AUC improves but PR-AUC falls. What might be happening?
**Answer:**
- Ranking behavior may have changed differently for the positive class.
- Class prevalence or sample composition may differ.
- Inspect precision-recall curve and confusion matrix at operating thresholds.
- Confirm both metrics use the same evaluation sample.

**Follow-up:** Which metric is more useful when churn is rare?

### Q7. Overall recall is good, but one contract segment has very low recall. What do you do?
**Answer:**
- Check segment sample size and label quality.
- Review feature distributions and errors.
- Assess business impact and governance requirements.
- Consider targeted data or model improvements and validate them.
- Do not hide the segment result behind the overall score.

**Follow-up:** How would you prevent overfitting to one segment?

### Q8. The final test score is lower than validation. Is the model broken?
**Answer:**
- Not necessarily; some variation is expected.
- Check whether the test population differs.
- Verify no leakage or repeated validation tuning occurred.
- Review sample size and confidence intervals.
- Report the gap honestly and investigate.

**Follow-up:** How could repeated validation tuning make validation optimistic?

## 3. Project questions

### Q9. How did you select the final candidate?
**Answer:**
- I compared candidates on a consistent validation strategy.
- I used metrics aligned with the churn use case.
- I considered threshold, capacity, segment behavior and operational complexity.
- I then evaluated the selected model on the untouched test set.

**Follow-up:** What would cause you to reject a model with the highest PR-AUC?

### Q10. Why include a business recommendation in a technical evaluation report?
**Answer:**
- Metrics need interpretation for the people using the model.
- A recommendation explains whether the model meets requirements and what limitations remain.
- It makes the decision auditable.
- It prevents technical scores from being mistaken for guaranteed business impact.

**Follow-up:** How would you measure actual retention uplift after launch?

## 4. Rapid revision
- TP: churner correctly flagged.
- FP: non-churner flagged.
- FN: churner missed.
- TN: non-churner correctly rejected.
- ROC-AUC: ranking discrimination.
- PR-AUC: precision-recall summary.
- Threshold: chosen using validation and business constraints.
- Final test: untouched until final evaluation.
