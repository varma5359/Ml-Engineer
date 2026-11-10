# Day 72: Interview Questions — Model Monitoring and Data Drift

**Phase:** 7 — Deployment and MLOps

## 1. Core questions

### Q1. Why monitor an ML model after deployment?
**Answer:**
- Production data and business conditions can change.
- Model quality can degrade over time.
- Monitoring detects data, prediction and service changes.
- It supports investigation, retraining and rollback decisions.

### Q2. What is data drift?
**Answer:**
- A change in the distribution of model input features.
- Example: customer ages or transaction amounts shift.
- It can indicate a change in the population or data pipeline.
- It does not automatically prove prediction quality has declined.

### Q3. What is concept drift?
**Answer:**
- A change in the relationship between features and the target.
- The same input patterns may lead to different outcomes over time.
- It requires labels or other outcome evidence to assess reliably.
- Retraining may help, but investigation is needed first.

### Q4. What is prediction drift?
**Answer:**
- A change in the distribution of model outputs.
- It may result from input changes, model changes or system issues.
- Monitor score and class distributions.
- Investigate rather than assuming it is a model failure.

### Q5. What is PSI?
**Answer:**
- Population Stability Index compares proportions across defined bins.
- It summarizes distribution change.
- It is a heuristic and depends on binning and sample size.
- It should be interpreted with other diagnostics.

### Q6. What is a reference dataset?
**Answer:**
- A baseline population used for comparison.
- It may be a training or approved baseline window.
- Its schema, time period and version should be recorded.
- It must be relevant to the production population.

## 2. Scenario questions

### Q7. Data drift is high but model accuracy is unchanged. What do you do?
**Answer:**
- Confirm labels and evaluation sample are reliable.
- Check which features shifted and why.
- Review performance by segment and time.
- Continue monitoring and document the risk.
- Do not retrain solely because one drift statistic is high.

### Q8. Accuracy drops but feature distributions look stable. What could explain it?
**Answer:**
- Concept drift.
- Label quality or definition changes.
- Evaluation population or measurement changes.
- Model-serving or feature transformation mismatch.
- Investigate delayed outcomes and pipeline lineage.

### Q9. Production data contains a new category. What should monitoring do?
**Answer:**
- Record unknown-category rate.
- Validate schema and encoding behavior.
- Determine whether the category is legitimate or a data defect.
- Decide whether mapping, feature update or retraining is required.

### Q10. How do you choose drift thresholds?
**Answer:**
- Start with domain and historical variability.
- Validate on known stable and changed periods.
- Consider sample size and false-alert cost.
- Use thresholds as investigation triggers, not universal truths.

## 3. Project questions

### Q11. Describe your Data Drift Detector.
**Answer:**
- I selected a versioned reference dataset.
- I compared production-like data with the same feature schema.
- I measured missingness and distribution statistics.
- I calculated a drift score and plotted key features.
- I generated an investigation report with threshold limitations.

### Q12. Does drift always require retraining?
**Answer:**
- No.
- The shift may be harmless, seasonal or caused by a data defect.
- Investigate feature meaning and actual performance.
- Retrain only when evidence and validation support it.

### Q13. What should a drift report contain?
**Answer:**
- Reference and production windows.
- Dataset versions and sample sizes.
- Schema and missing-value checks.
- Per-feature drift statistics and plots.
- Thresholds, warnings and investigation notes.
- Limitations and recommended next checks.

## 4. Rapid revision
- Data drift: input distribution change.
- Concept drift: input-target relationship change.
- Prediction drift: output distribution change.
- PSI: binned distribution comparison.
- Drift alert: investigate, not automatic retraining.
