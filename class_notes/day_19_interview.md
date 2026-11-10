# Day 19: Interview Questions — Linear Regression, Assumptions and Equation

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. What is regression?
**Answer:** Regression is a supervised-learning task that predicts a numerical target, such as house price, delivery duration or monthly demand.
**Follow-up:** What is the difference between regression and classification?  
**Answer:** Regression predicts a numerical quantity; classification predicts a category or class.

### Q2. What is linear regression?
**Answer:** It predicts a numerical target by combining input features with learned weights and an intercept. In a simple one-feature case, it fits a straight-line relationship.
**Follow-up:** Does the line need to pass through every training observation?  
**Answer:** No. It represents the fitted relationship and generally minimizes a selected training objective.

### Q3. What is a coefficient?
**Answer:** A learned weight describing how the model's prediction changes with a feature while other included features are held constant.
**Follow-up:** Does a positive coefficient prove causation?  
**Answer:** No. It describes an association within the fitted model.

### Q4. What is an intercept?
**Answer:** It is the model's predicted starting value when input features are zero. It may not have practical meaning if zero is outside the observed data range.
**Follow-up:** Can the intercept be negative?  
**Answer:** Yes. Its value depends on the data and fitted relationship; judge whether it is meaningful in context.

## Practical and scenario questions

### Q5. How do you train a LinearRegression model in scikit-learn?
**Answer:** Create X as a two-dimensional feature table and y as the numerical target, split them appropriately, initialize LinearRegression, call fit on training data and predict on held-out or new data.
**Follow-up:** What does fit() do?  
**Answer:** It learns the model parameters from the training features and target.

### Q6. A house-price model predicts premium properties too low. What do you inspect?
**Answer:** Inspect residuals by price band, check whether premium properties are represented in training data, identify missing features and assess whether a straight-line model is adequate.
**Follow-up:** Should you immediately use a more complex algorithm?  
**Answer:** Not before understanding the error pattern and validating that added complexity improves performance.

### Q7. A coefficient is positive. Can the business claim that increasing the feature will cause the target to increase?
**Answer:** No. A coefficient describes a fitted association. Causal claims require appropriate evidence and study design.
**Follow-up:** What if the model predicts accurately?  
**Answer:** Predictive accuracy still does not establish causality.

### Q8. Training predictions are accurate but test predictions are poor. What could be happening?
**Answer:** Overfitting, data distribution mismatch, leakage, an unsuitable split or a model that does not generalize.
**Follow-up:** What should you inspect first?  
**Answer:** Confirm the split and leakage controls, then compare training and test errors and data distributions.

### Q9. A coefficient changes significantly when a correlated feature is added. What may explain it?
**Answer:** Multicollinearity or overlapping information among predictors can make individual coefficients unstable.
**Follow-up:** Does that automatically mean the model cannot predict?  
**Answer:** No. It may mainly affect coefficient interpretation and stability.

### Q10. The model predicts for an area far beyond every training example. What is the concern?
**Answer:** Extrapolation. The learned relationship may not continue outside the range represented in training data.
**Follow-up:** What should an API do?  
**Answer:** Validate or flag inputs outside the supported range according to the application's policy.

## Residual and diagnostics questions

### Q11. What is a residual?
**Answer:** Actual target minus predicted target for one observation. Positive residuals indicate underprediction; negative residuals indicate overprediction.
**Follow-up:** Why plot residuals against predictions?  
**Answer:** To find patterns such as curvature, changing spread or systematic bias.

### Q12. What assumptions or diagnostics do you consider?
**Answer:** Check whether the relationship is reasonably represented by the model, whether observations are independent for the task, whether residual spread changes, and whether influential observations or multicollinearity affect interpretation. Assumptions for classical statistical inference are not all universal requirements for prediction.
**Follow-up:** Is normal residual distribution always required for useful predictions?  
**Answer:** No. It matters for some classical inference procedures, not as a universal requirement for predictive use.

**Senior developer principle:** A fitted coefficient is not a causal explanation. Validate the model's errors and the meaning of its features.
