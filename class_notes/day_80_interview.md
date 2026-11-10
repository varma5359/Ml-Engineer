# Day 80: Interview Questions — Capstone Baseline Model and Initial Experiments

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. What is a baseline model?
**Answer:**
- A simple reference for evaluating more advanced models.
- It provides a minimum performance comparison.
- It helps identify whether model complexity adds value.
- Baseline choice should match the task.

**Follow-up:** Why might a majority-class baseline be misleading for churn?

### Q2. Why use DummyClassifier?
**Answer:**
- It creates predictions using simple strategies.
- It does not learn complex relationships from features.
- It gives a reference score.
- It can reveal when a model is not outperforming a trivial strategy.

**Follow-up:** Which DummyClassifier strategies could you compare?

### Q3. Why is logistic regression a useful classification baseline?
**Answer:**
- It is relatively simple and interpretable.
- It produces class probabilities.
- It is efficient for many tabular problems.
- It provides a useful reference before trying more complex models.

**Follow-up:** What assumptions or limitations should you consider?

### Q4. Explain precision and recall for churn.
**Answer:**
- Precision: among flagged customers, the proportion who actually churn.
- Recall: among actual churners, the proportion identified.
- Higher recall can find more churners but may increase outreach volume.
- Higher precision can reduce wasted outreach but may miss more churners.

**Follow-up:** How does review capacity influence your preferred trade-off?

### Q5. Why preserve the test set?
**Answer:**
- It estimates performance on data not used for model selection.
- Repeated tuning on test results creates optimistic bias.
- Use validation data for iterative choices.
- Evaluate the final selected model on the test set once the workflow is fixed.

**Follow-up:** How would you handle repeated time-based snapshots?

## 2. Scenario questions

### Q6. DummyClassifier has 80% accuracy and logistic regression has 79%. Is the dummy better?
**Answer:**
- Not necessarily.
- Inspect churn recall, precision, F1 and ranking metrics.
- The majority-class model may never identify churn.
- Consider the business objective and error costs.

**Follow-up:** What would the confusion matrix reveal?

### Q7. ROC-AUC is high but recall at the chosen threshold is low. Why?
**Answer:**
- ROC-AUC evaluates ranking across thresholds.
- The current threshold may be too high for the desired recall.
- Inspect the precision-recall curve and threshold behavior.
- Choose a threshold using validation data and business constraints.

**Follow-up:** What happens to precision when the threshold is lowered?

### Q8. Your metric function fails because only one class is present. What do you check?
**Answer:**
- Verify the split and target distribution.
- Check stratification or split design.
- Confirm label mapping and filtering.
- Some metrics such as ROC-AUC require both classes.

**Follow-up:** What alternative evaluation approach is suitable for very small datasets?

## 3. Project questions

### Q9. What should a baseline report contain?
**Answer:**
- Dataset and split information.
- Target and positive-class definition.
- Model configurations.
- Metrics and confusion matrices.
- Comparison with the dummy baseline.
- Limitations and next experiments.

**Follow-up:** How would you make the report reproducible?

### Q10. How do you know whether your baseline is useful?
**Answer:**
- It provides a transparent reference.
- It should be evaluated using appropriate metrics.
- It must be compared on the same data as candidate models.
- It should connect to the problem's business objective.

**Follow-up:** What would you do if the baseline already meets the business requirement?

## 4. Rapid revision
- Baseline: simple reference model.
- DummyClassifier: trivial prediction strategy.
- Precision: correctness among flagged positives.
- Recall: fraction of actual positives detected.
- F1: harmonic mean of precision and recall.
- False negative: missed churner.
- Validation: guides model choices.
- Test: final unbiased estimate.
