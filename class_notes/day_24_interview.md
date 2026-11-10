# Day 24: Interview Questions — Regularization: Ridge and Lasso

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is regularization?
**Answer:** It adds a penalty to discourage excessively large model coefficients. It can reduce complexity and improve generalization when selected appropriately.
**Follow-up:** Does regularization always improve performance?  
**Answer:** No. Too much penalty can cause underfitting.

### Q2. What is Ridge regression?
**Answer:** Ridge uses an L2 penalty and generally shrinks coefficients towards zero without usually setting them exactly to zero.
**Follow-up:** When might Ridge be useful?  
**Answer:** When many predictors contain useful information or are correlated and coefficient magnitude needs controlling.

### Q3. What is Lasso regression?
**Answer:** Lasso uses an L1 penalty and can set some coefficients exactly to zero, creating a sparse coefficient representation.
**Follow-up:** Does a zero coefficient prove the feature is universally useless?  
**Answer:** No. It depends on the dataset, alpha and other predictors.

### Q4. What does alpha control?
**Answer:** Alpha controls regularization strength. Larger alpha generally means a stronger penalty.
**Follow-up:** What can happen if alpha is too large?  
**Answer:** The model may become too simple and underfit.

### Q5. Why scale features before Ridge or Lasso?
**Answer:** The penalty acts on coefficient sizes, and feature units affect those sizes. Scaling makes the penalty more comparable across numerical inputs.
**Follow-up:** How do you prevent scaling leakage?  
**Answer:** Fit the scaler inside a pipeline using training data only.

## Scenario questions

### Q6. Coefficients are very large and unstable. What can you try?
**Answer:** Check feature scaling and multicollinearity, then compare regularized models such as Ridge or Lasso using validation.
**Follow-up:** Should you automatically choose Lasso?  
**Answer:** No. Compare alternatives and the modelling objective.

### Q7. Lasso sets a feature coefficient to zero. What does it mean?
**Answer:** That fitted Lasso model does not use the feature under the selected data and alpha. It does not prove the feature has no value in every population or model.
**Follow-up:** Could the selected feature set change with another sample?  
**Answer:** Yes, especially when predictors are correlated or data is limited.

### Q8. Should alpha be chosen using final test performance?
**Answer:** No. Use validation or cross-validation to choose alpha and reserve the test set for final evaluation.
**Follow-up:** Why use GridSearchCV around a pipeline?  
**Answer:** It lets each fold fit its own scaler and model, reducing preprocessing leakage during parameter selection.

### Q9. Ridge performs better than Lasso on validation data. What might that suggest?
**Answer:** Retaining many predictors with smaller coefficients may suit the problem better than a sparse representation.
**Follow-up:** Is this guaranteed to remain true in production?  
**Answer:** No. Validate on representative data and monitor performance.

### Q10. A regularized model performs worse than ordinary regression. What do you investigate?
**Answer:** Alpha strength, scaling, feature quality, sample size and validation stability. The penalty may be too strong or unnecessary for the task.

## Practical questions

### Q11. What is the difference between Ridge and Lasso penalties?
**Answer:** Ridge uses L2 regularization and generally shrinks coefficients. Lasso uses L1 regularization and can set coefficients exactly to zero.

### Q12. Why use cross-validation for alpha selection?
**Answer:** It compares candidate penalty strengths across multiple training/validation folds and reduces reliance on one split.
**Follow-up:** Should test data be included in alpha selection?  
**Answer:** No.

**Senior developer principle:** Regularization is a controlled trade-off between model flexibility and stability, not a switch that automatically improves every model.
