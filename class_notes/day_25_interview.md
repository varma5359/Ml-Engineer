# Day 25: Interview Questions — Logistic Regression and Sigmoid Function

**Pathway:** Machine Learning Engineer

## Core concepts

### Q1. Why is logistic regression used for classification?
**Answer:** It estimates class probabilities by transforming a model score into a value between zero and one, then assigns a class using a decision threshold.
**Follow-up:** Why does its name include regression?  
**Answer:** It models a score using a regression-style structure, but it is commonly used for classification.

### Q2. What is the sigmoid function?
**Answer:** It maps a real-valued score into a value between zero and one. In binary logistic regression, the output is used as an estimated positive-class probability.
**Follow-up:** Does a probability of 0.90 guarantee the outcome?  
**Answer:** No. It is an estimate, not a guarantee.

### Q3. What is the difference between predict() and predict_proba()?
**Answer:** predict() returns class labels. predict_proba() returns probability estimates for each class.
**Follow-up:** Why inspect classes_?  
**Answer:** To confirm the order of the probability columns.

### Q4. What does max_iter control?
**Answer:** It sets the maximum number of optimization iterations for the estimator.
**Follow-up:** If a convergence warning appears, what do you inspect?  
**Answer:** Feature scaling, data quality, model settings and the iteration limit.

## Scenario questions

### Q5. A customer has a churn probability of 0.82. Does that guarantee churn?
**Answer:** No. It is an estimated probability. Its reliability depends on model performance and calibration.
**Follow-up:** What does calibration mean?  
**Answer:** Predicted probabilities should correspond reasonably well to observed event frequencies.

### Q6. The model predicts every customer as non-churn. What do you inspect?
**Answer:** Class balance, threshold, confusion matrix, recall, feature quality and whether the model learned useful patterns.
**Follow-up:** Why can accuracy hide this failure?  
**Answer:** If non-churn is the majority class, predicting it for every row can still achieve high accuracy.

### Q7. A new contract category appears in production. What should happen?
**Answer:** Use a documented unknown-category policy in the fitted encoder and monitor the new category.
**Follow-up:** Does ignoring the category mean the model learned its effect?  
**Answer:** No. It only avoids a transformation error.

### Q8. Probability ranking is useful but estimates are overconfident. What do you investigate?
**Answer:** Probability calibration, evaluation data representativeness and whether calibration methods are appropriate.
**Follow-up:** Should calibration be assessed on training data?  
**Answer:** No. Use suitable held-out or cross-validation procedures.

### Q9. Why separate the target from the features?
**Answer:** The target is the outcome being predicted. Including it in X would let the model access the answer and create leakage.

## Code interpretation

### Q10. What does predict_proba(X_test) return?
**Answer:** A probability for each class for each test observation.
**Follow-up:** How do you select the positive-class column?  
**Answer:** Inspect classes_ and select the matching column.

### Q11. Why use a Pipeline for logistic regression?
**Answer:** It combines preprocessing and the classifier so the same fitted transformations are used during training and prediction.

**Senior developer principle:** A probability is an estimate that must be evaluated and communicated responsibly.
