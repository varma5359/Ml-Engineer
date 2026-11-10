# Day 20: Interview Questions — Cost Functions and Residual Analysis

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is a residual?
**Answer:** A residual is the actual target minus the model prediction for one observation. Positive means underprediction; negative means overprediction.
**Follow-up:** Why is the sign useful?  
**Answer:** It reveals the direction of prediction error, not just its magnitude.

### Q2. What is a loss function?
**Answer:** A loss function measures how wrong a prediction or set of predictions is. Training algorithms use an objective to adjust model parameters and reduce loss.
**Follow-up:** Do all loss functions penalize errors in the same way?  
**Answer:** No. Some give larger errors more influence than others.

### Q3. What is MAE?
**Answer:** Mean Absolute Error is the average absolute prediction error. It is expressed in the same unit as the target.
**Follow-up:** Why is it easy to communicate?  
**Answer:** A stakeholder can interpret it as a typical absolute error in minutes, price units or another target unit.

### Q4. What is MSE?
**Answer:** Mean Squared Error averages squared prediction errors. It is in squared target units and gives large errors more influence.
**Follow-up:** Why might MSE be sensitive to one large error?  
**Answer:** Squaring increases the contribution of larger residuals.

### Q5. What is RMSE?
**Answer:** Root Mean Squared Error is the square root of MSE. It returns the result to the same unit as the target while still emphasizing large errors.
**Follow-up:** Why can RMSE exceed MAE?  
**Answer:** Large errors have extra influence in the squared-error calculation.

### Q6. What does R-squared describe?
**Answer:** It compares model performance with a mean-target reference. A negative value on evaluation data means the model is worse than that reference.
**Follow-up:** Does high R-squared guarantee acceptable business errors?  
**Answer:** No. Inspect error magnitude, segments and residual patterns.

## Scenario questions

### Q7. A delivery model has MAE of five minutes. How do you explain it?
**Answer:** On the evaluated observations, the average absolute difference between predicted and actual delivery time is five minutes.
**Follow-up:** Does every prediction have a five-minute error?  
**Answer:** No. Some errors may be smaller and others larger.

### Q8. RMSE is much larger than MAE. What might this indicate?
**Answer:** A few large prediction errors may be contributing strongly to RMSE.
**Follow-up:** What would you inspect?  
**Answer:** Sort observations by absolute residual and investigate the largest errors.

### Q9. A model has good overall MAE but consistently underpredicts luxury houses. What do you do?
**Answer:** Inspect residuals by price segment, sample representation, feature coverage and source quality. Validate any change on appropriate held-out data.
**Follow-up:** Why report segment counts?  
**Answer:** A metric based on very few examples can be unstable.

### Q10. Can R-squared be negative on test data?
**Answer:** Yes. It means the model performs worse than the mean-target reference under the R-squared comparison.
**Follow-up:** Should you hide the negative result?  
**Answer:** No. Investigate the model and report it transparently.

### Q11. Which metric should be used if very large errors are especially costly?
**Answer:** Consider RMSE or another loss that penalizes large errors, but select the metric based on the actual business costs and inspect the full error distribution.
**Follow-up:** Is RMSE always the best choice?  
**Answer:** No. It depends on the problem and cost structure.

### Q12. Why inspect residuals by region or product category?
**Answer:** An overall metric can hide a weak segment or systematic underprediction/overprediction. Segment-level analysis shows where the model is unreliable.

## Code interpretation

### Q13. What does mean_absolute_error(y_true, y_pred) return?
**Answer:** The average absolute difference between the aligned actual and predicted values.
**Follow-up:** What must be true about the arrays?  
**Answer:** They must refer to the same observations in the same order.

### Q14. Why does MSE use squared errors?
**Answer:** It makes errors non-negative and gives larger errors more influence.
**Follow-up:** What is one drawback?  
**Answer:** It is sensitive to large errors and is measured in squared target units.

**Senior developer principle:** One metric cannot explain every model failure. Combine error summaries, residual plots and business-segment analysis.
