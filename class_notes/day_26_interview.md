# Day 26: Interview Questions — Decision Boundaries and Classification Thresholds

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is a decision boundary?
**Answer:** It is the separation in feature space between regions that a classifier assigns to different classes.
**Follow-up:** Is every decision boundary a straight line?  
**Answer:** No. It depends on the model and feature representation.

### Q2. What is a classification threshold?
**Answer:** It is the cut-off applied to a score or probability to convert it into a final class label.
**Follow-up:** Is 0.50 always the correct threshold?  
**Answer:** No. Business costs and operational needs may require a different threshold.

### Q3. What happens when the threshold is lowered?
**Answer:** The model generally predicts more positive cases. Recall may increase, but false positives can also increase and precision may fall.
**Follow-up:** What should you monitor?  
**Answer:** Precision, recall, false positives, false negatives and alert volume.

### Q4. What is the difference between a model probability and a predicted class?
**Answer:** A probability is a confidence-like estimate for a class; a predicted class is the final label after applying a decision rule or threshold.
**Follow-up:** Can two customers with different probabilities receive the same class?  
**Answer:** Yes, if both probabilities fall on the same side of the threshold.

## Scenario questions

### Q5. Fraud is expensive to miss. What should threshold selection consider?
**Answer:** Recall, false-negative cost, precision and the team's investigation capacity.
**Follow-up:** Should the threshold simply be set very low?  
**Answer:** No. That may create too many false alerts and overwhelm the review team.

### Q6. A review team can investigate only 100 alerts per day. What should you consider?
**Answer:** Alert volume at different thresholds, ranking quality, the value of investigated cases and operational capacity.
**Follow-up:** Why report positive prediction count?  
**Answer:** It connects the threshold to workload.

### Q7. The default threshold gives high accuracy but low recall. What might you try?
**Answer:** Evaluate lower thresholds on validation data and compare precision, recall, F1 and operational costs.
**Follow-up:** Should you choose based on the test score?  
**Answer:** No. Use validation data and keep test data for final evaluation.

### Q8. Can you select the final threshold on the test set?
**Answer:** No. Threshold selection is a development decision. The test set should remain independent.
**Follow-up:** What should happen after threshold selection?  
**Answer:** Apply it once to test probabilities and report final held-out metrics.

### Q9. A model's probability estimates are poorly calibrated. What is the concern?
**Answer:** The probability values may not correspond to observed event frequencies, so a threshold may not behave as expected.
**Follow-up:** What should you assess?  
**Answer:** Calibration on appropriate held-out data and whether a calibration method is suitable.

## Code interpretation

### Q10. What does (probabilities >= threshold).astype(int) do?
**Answer:** It converts each probability into a binary label based on the threshold.
**Follow-up:** What happens when probability equals the threshold?  
**Answer:** It is assigned to the positive class because the comparison uses greater-than-or-equal.

### Q11. Why visualize a decision boundary?
**Answer:** It helps show how the classifier separates feature space and how a threshold change affects the assigned regions.

**Senior developer principle:** Threshold choice is a business decision informed by validation evidence, not a default constant to accept blindly.
