# Day 55: Interview Questions — XGBoost and Practical Boosting Workflows

**Phase:** 6 — Advanced ML and Practical Applications

## 1. Core questions

### Q1. What is XGBoost?
**Answer:**
- XGBoost stands for Extreme Gradient Boosting.
- It is a library for efficient gradient-boosted decision trees.
- It includes regularization and computational optimizations.
- It supports classification and regression objectives.

### Q2. How does XGBoost control overfitting?
**Answer:**
- L1 and L2 regularization control model complexity.
- Tree depth and child-weight constraints limit splits.
- Row and column sampling add randomness.
- Learning rate controls each stage's contribution.
- Validation helps select settings.

### Q3. Explain learning_rate and n_estimators.
**Answer:**
- Learning rate scales each boosting round.
- n_estimators sets the number of rounds.
- A lower learning rate may need more rounds.
- Tune both using validation data.

### Q4. What is scale_pos_weight?
**Answer:**
- It adjusts the relative weight of the positive class in binary classification.
- It can help when positive examples are less frequent.
- Negative-count divided by positive-count is a common starting heuristic.
- Validate the impact on precision, recall and business costs.

### Q5. Why is PR-AUC useful for churn?
**Answer:**
- Churn may be the minority class.
- Precision measures how many predicted churners are actual churners.
- Recall measures how many actual churners are identified.
- The precision-recall curve focuses on positive-class retrieval.

### Q6. Compare predict and predict_proba.
**Answer:**
- predict returns class labels.
- predict_proba returns class probability estimates when supported.
- A threshold converts probability estimates to labels.
- Choose the threshold using validation data and business costs.

## 2. Scenario questions

### Q7. Accuracy is 92%, but churn recall is 20%. Is the model good?
**Answer:**
- Not necessarily; it misses most churners.
- Inspect class balance and confusion matrix.
- Evaluate precision, recall, F1 and PR-AUC.
- Investigate class weights, data quality and threshold.
- Discuss retention capacity and false-positive cost.

### Q8. Training performance is much better than validation performance. What do you do?
**Answer:**
- Check leakage and split design.
- Reduce max_depth or increase min_child_weight.
- Tune learning rate, estimators and regularization.
- Consider row and column sampling.
- Use cross-validation and compare a baseline.

### Q9. The business can contact only 100 customers. How would you use the model?
**Answer:**
- Generate risk scores for eligible customers.
- Rank by estimated risk.
- Validate ranking and probability quality.
- Consider campaign costs and eligibility.
- Monitor actual outcomes and avoid treating scores as certainty.

### Q10. How do you prevent leakage in churn prediction?
**Answer:**
- Define prediction timestamp and outcome window.
- Exclude information known only after that timestamp.
- Use an appropriate time-based split if required.
- Fit preprocessing on training data only.
- Keep test data isolated.

## 3. Project questions

### Q11. Explain your churn boosting workflow.
**Answer:**
1. Define churn and prediction horizon.
2. Inspect data and class balance.
3. Establish a baseline.
4. Create a leakage-safe split.
5. Train and tune XGBoost.
6. Evaluate class-specific metrics.
7. Select a threshold using validation.
8. Document limitations.

### Q12. Does feature importance prove why a customer churns?
**Answer:**
- No. It describes model reliance under a chosen calculation.
- It does not establish causation.
- Correlated features complicate interpretation.
- Use suitable local explanation and domain review.

### Q13. What is early stopping?
**Answer:**
- Training monitors a validation metric.
- It stops after the metric fails to improve for a configured number of rounds.
- It can avoid unnecessary boosting rounds.
- The final test set should not be used as the early-stopping validation set.

### Q14. What checks are needed before deployment?
**Answer:**
- Input schema and feature availability.
- Model and preprocessing versions.
- Performance and threshold policy.
- Latency and failure handling.
- Access controls and privacy.
- Drift monitoring and rollback.

## 4. Rapid revision
- XGBoost: optimized boosted trees.
- Regularization: controls complexity.
- scale_pos_weight: positive-class weighting.
- PR-AUC: useful for minority positive class.
- Threshold: selected using validation and business costs.
- Early stopping: monitors validation performance.
- Importance: not causation.
