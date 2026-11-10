# Day 23: Interview Questions — Polynomial Regression and Nonlinear Relationships

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. Why might a straight-line model be insufficient?
**Answer:** The relationship between the feature and target may be curved or nonlinear, so a straight line may miss the pattern.
**Follow-up:** How can you notice this?  
**Answer:** Inspect scatter plots and residual plots for systematic curvature.

### Q2. What is polynomial regression?
**Answer:** It creates transformed features such as squared or cubed terms and fits a regression model on those features. This lets the model represent curved relationships in the original feature space.
**Follow-up:** Is it still linear in the learned coefficients?  
**Answer:** Yes. The transformed features are combined linearly by the regression estimator.

### Q3. What does polynomial degree control?
**Answer:** It controls the complexity of the generated feature terms. Higher degree can represent more complex curves but can also fit noise.
**Follow-up:** Is higher degree always better?  
**Answer:** No. Choose complexity using validation or cross-validation.

## Scenario questions

### Q4. A linear model shows a curved residual pattern. What could you test?
**Answer:** Polynomial features or another nonlinear model, then compare performance using validation.
**Follow-up:** Should you select a model based only on training error?  
**Answer:** No. Training error alone can reward overfitting.

### Q5. A degree-12 model fits training data almost perfectly but fails validation. What is likely?
**Answer:** Overfitting. The model may have learned noise or sample-specific patterns.
**Follow-up:** What would you try?  
**Answer:** Reduce degree, add regularization or compare a simpler model using cross-validation.

### Q6. Is polynomial regression reliable for predictions far beyond the training range?
**Answer:** Often not. A polynomial curve can behave unpredictably outside the observed range.
**Follow-up:** What should you check?  
**Answer:** Whether new inputs lie within the supported feature range and whether domain knowledge supports extrapolation.

### Q7. Degree two improves validation error slightly but degree five improves training error greatly. Which do you prefer?
**Answer:** Prefer the simpler model if validation performance is comparable and it is more stable and easier to support.
**Follow-up:** Why?  
**Answer:** Added complexity should provide measurable generalization benefit.

## Practical questions

### Q8. Why use PolynomialFeatures inside a Pipeline?
**Answer:** It keeps feature generation and regression together and ensures the transformer is fitted within the training workflow.
**Follow-up:** What happens during predict?  
**Answer:** The pipeline applies the fitted polynomial transformation before passing the result to the regression model.

### Q9. How do you compare different polynomial degrees?
**Answer:** Evaluate candidate degrees using validation or cross-validation and inspect error stability and residual patterns.
**Follow-up:** Why not repeatedly choose degree using the final test score?  
**Answer:** That makes the test set part of model selection.

### Q10. What is one risk of polynomial feature expansion?
**Answer:** The number and scale of generated features can grow quickly, increasing complexity and possible numerical instability.

**Senior developer principle:** A curve that fits the training data beautifully may still be a poor model for new observations.
