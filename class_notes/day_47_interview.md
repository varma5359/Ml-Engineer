# Day 47: Interview Questions — Regression Evaluation

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is MAE?
**Answer:**
- Mean Absolute Error is the average absolute difference between actual and predicted values.
- It is expressed in the target's original unit.
- It is easy to explain to business users.
- It is less sensitive to large errors than squared-error metrics.

**Follow-up:** When would you prefer MAE?  
**Answer:** When a typical absolute error is important and we do not want large errors to dominate as strongly as they do under squared-error metrics.

### Q2. What is MSE?
**Answer:**
- Mean Squared Error averages squared prediction errors.
- Large errors receive much more influence.
- It is useful when large mistakes should be penalized strongly.
- Its unit is the square of the target unit, so it is less intuitive for business communication.

**Follow-up:** What is a limitation of MSE?  
**Answer:** It is sensitive to outliers and can be dominated by a few very large errors.

### Q3. What is RMSE?
**Answer:**
- Root Mean Squared Error is the square root of MSE.
- It is expressed in the target's original unit.
- It penalizes large errors more strongly than MAE.
- It is often easier to explain than MSE.

**Follow-up:** Why can RMSE be much larger than MAE?  
**Answer:** A few unusually large errors may have a strong influence on RMSE.

### Q4. What is R² score?
**Answer:**
- R² compares the model's predictions with a simple baseline that predicts the average target value.
- A value near 1 indicates that the model explains much of the observed target variation.
- A value near 0 indicates performance similar to the mean baseline.
- A negative value means the model performs worse than that baseline on the evaluated data.

**Follow-up:** Is R² the percentage of predictions that are correct?  
**Answer:** No. Regression predicts continuous values, and R² is not classification accuracy.

### Q5. Why use MAE and RMSE together?
**Answer:**
- MAE describes the average absolute error.
- RMSE gives more influence to large errors.
- Comparing them helps identify whether a small number of large mistakes may be affecting the model.
- Review residuals to understand the cause.

### Q6. Can R² be negative?
**Answer:**
- Yes, especially on unseen test data.
- It means the model performs worse than predicting the mean target value for every observation in that evaluation set.
- Investigate overfitting, distribution shift, target alignment and preprocessing.

## 2. Metric selection and comparison

### Q7. Which metric would you use for house price prediction?
**Answer:**
- MAE is easy to communicate in currency units.
- RMSE is useful if very large price errors are especially costly.
- R² gives a relative comparison with the mean baseline.
- Inspect errors across low-, medium- and high-price properties.

### Q8. A model has lower MAE but higher RMSE than another model. How do you decide?
**Answer:**
- The first model may have smaller typical absolute errors but some larger mistakes.
- Inspect residual distributions and high-error records.
- Understand whether large errors have special business costs.
- Select the model based on the business objective, not one metric alone.

### Q9. Why is MSE less intuitive for business users?
**Answer:**
- It is expressed in squared target units.
- A value such as 400 square units is harder to interpret than an error in the original unit.
- RMSE or MAE is often easier to communicate.

### Q10. Is a high R² sufficient to approve a regression model?
**Answer:**
- No.
- Review MAE and RMSE in business units.
- Inspect residual patterns and important segments.
- Compare with a baseline.
- Check that the test data represents the intended use.

## 3. Troubleshooting questions

### Q11. RMSE is much higher than MAE. What do you investigate?
**Answer:**
- Identify records with the largest residuals.
- Check data entry, target quality and unusual operating conditions.
- Review whether errors are concentrated in a particular segment.
- Determine whether large errors are valid rare events or data issues.

### Q12. Training R² is high but test R² is negative. What might be wrong?
**Answer:**
- The model may be overfitting.
- Train and test populations may differ.
- Preprocessing or target alignment may be incorrect.
- Compare with a baseline and inspect validation results.

### Q13. Why should residuals be plotted?
**Answer:**
- They can reveal systematic overprediction or underprediction.
- Patterns may indicate missing relationships or unsuitable model assumptions.
- Changing spread can indicate that error size differs across prediction ranges.
- A single summary metric can hide these patterns.

### Q14. Why must actual and predicted values be aligned row by row?
**Answer:**
- Each prediction must correspond to the correct actual observation.
- If order is mismatched, residuals and all metrics become invalid.
- Preserve identifiers or indices when joining predictions to actual values.

## 4. Scenario-based interview questions

### Scenario 1: RMSE is much higher than MAE
**Expected answer:**
- A small number of large errors may be influencing RMSE.
- Inspect residuals and records with largest absolute errors.
- Check data quality, target distribution and subgroups.
- Do not remove unusual observations without a valid reason.

### Scenario 2: R² is negative on test data
**Expected answer:**
- The model performs worse than the mean-target baseline on this test set.
- Check train-test distribution differences, preprocessing, target alignment and overfitting.
- Compare with a simple baseline and review residuals.

### Scenario 3: Model A has lower MAE but Model B has lower RMSE
**Expected answer:**
- Model A may have smaller typical absolute errors.
- Model B may avoid some large errors or distribute errors differently.
- Inspect residuals and the business cost of large mistakes.
- Choose the metric aligned with the use case and validate by segment.

### Scenario 4: A model has high R² but poor business usefulness
**Expected answer:**
- Check MAE and RMSE in business units.
- Inspect errors for important high-value or low-value segments.
- Check systematic overprediction or underprediction.
- Confirm that evaluation data represents the real decision.

## 5. Strong points to remember in interviews

- MAE: average absolute error in original target units.
- MSE: squared errors; large mistakes have greater influence.
- RMSE: square root of MSE; original target units.
- R²: comparison with mean-target baseline, not accuracy.
- Negative R² is possible on test data.
- A high R² does not guarantee acceptable errors for every segment.
- Compare models on the same evaluation observations.
- Inspect residuals instead of trusting one metric.

**Senior developer principle:** Translate regression metrics into the business unit and investigate the records that create the largest errors.
