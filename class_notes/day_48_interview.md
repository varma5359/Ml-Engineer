# Day 48: Interview Questions — Cross-Validation and Hyperparameter Tuning

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering

## 1. Core interview questions

### Q1. What is cross-validation?
**Answer:**
- A model evaluation approach that repeats training and validation on different folds.
- It reduces dependence on one particular train-validation split.
- It provides fold-level performance and an overall summary.
- It is commonly used for model comparison and hyperparameter selection.

**Follow-up:** Does cross-validation remove the need for a test set?  
**Answer:** No. Keep an untouched test set for final evaluation after model selection.

### Q2. Explain K-Fold cross-validation.
**Answer:**
- Divide the development data into K folds.
- Train on K-1 folds and validate on the remaining fold.
- Repeat so each fold is used for validation once.
- Summarize the fold scores and their variation.

**Follow-up:** What happens when K is increased?  
**Answer:** Each model fit uses a larger share of data, but the process may require more computation. The best K depends on sample size and practical constraints.

### Q3. What is Stratified K-Fold?
**Answer:**
- A classification splitting strategy.
- It tries to preserve class proportions in each fold.
- It is useful when the target classes are imbalanced.
- It helps avoid folds that contain too few minority-class examples.

### Q4. When should you avoid ordinary shuffled K-Fold?
**Answer:**
- For time-series forecasting where time order matters.
- For grouped data where records from the same person, device or organization must not leak across folds.
- For repeated measurements where related records are not independent.
- Choose a split strategy that reflects how the model will be used.

### Q5. What is hyperparameter tuning?
**Answer:**
- Searching for model configuration settings that improve validation performance.
- Hyperparameters are selected before or during model selection rather than learned as ordinary model parameters.
- Examples include tree depth, number of neighbours and regularization strength.

### Q6. What is the difference between model parameters and hyperparameters?
**Answer:**
- Model parameters are learned from training data, such as linear regression coefficients.
- Hyperparameters configure the learning process or model structure, such as maximum tree depth.
- Tuning searches for useful hyperparameter settings using validation evidence.

## 2. Grid search and randomized search

### Q7. What is GridSearchCV?
**Answer:**
- It evaluates every combination in a specified hyperparameter grid.
- Each combination is evaluated using cross-validation.
- It is useful when the search space is small and focused.
- It can become expensive when many parameters and values are included.

### Q8. What is RandomizedSearchCV?
**Answer:**
- It samples a fixed number of parameter combinations from supplied distributions or lists.
- It can explore a large search space with a limited compute budget.
- It does not guarantee that every combination or the global best setting is evaluated.

### Q9. GridSearchCV versus RandomizedSearchCV — when would you use each?
**Answer:**
- Use GridSearchCV for a small, carefully selected parameter grid.
- Use RandomizedSearchCV for a broader or expensive search space.
- Consider compute budget, model training time and the importance of each hyperparameter.
- Validate the selected model on untouched data.

## 3. Leakage and validation questions

### Q10. Why should scaling happen inside a cross-validation Pipeline?
**Answer:**
- A scaler learns information such as means and standard deviations from the data used to fit it.
- If it is fitted before splitting folds, validation information can influence preprocessing.
- A Pipeline fits the scaler separately on each training fold.
- The corresponding validation fold is transformed using only that fold's training information.

### Q11. Why should the test set not be used for hyperparameter tuning?
**Answer:**
- Tuning decisions would indirectly use information from the test set.
- Repeated selection can make the test result optimistic.
- Use cross-validation or validation data for model selection.
- Reserve the final test set for one final evaluation.

### Q12. Cross-validation scores vary widely. What does it indicate?
**Answer:**
- Performance may depend on which observations are used.
- The dataset may be small or contain difficult subgroups.
- The split strategy may not match the data structure.
- Inspect fold-level results, class balance, outliers and group/time dependencies.

### Q13. Why might negative MAE scores appear in scikit-learn?
**Answer:**
- Scikit-learn follows a convention that larger scoring values are better.
- Error metrics are therefore exposed as negative scores in some scoring names.
- Convert the values back to positive errors when presenting them to stakeholders.

## 4. Scenario-based interview questions

### Scenario 1: One train-test split gives a very different score from another
**Expected answer:**
- Performance may depend on which observations enter each split.
- Use cross-validation to examine performance across folds.
- Check whether the data contains groups, time order or class imbalance that requires a special split.

### Scenario 2: Cross-validation scores vary widely
**Expected answer:**
- Inspect fold-level results.
- Check sample size, outliers, class distribution and subgroup differences.
- Verify that splitting matches the data-generating process.
- Report variability rather than only the average.

### Scenario 3: A developer scales all rows before cross-validation
**Expected answer:**
- This can leak information from validation folds into preprocessing.
- Put the scaler inside a Pipeline.
- Let cross-validation fit preprocessing on each training fold only.

### Scenario 4: The team uses GridSearchCV with thousands of combinations
**Expected answer:**
- The search may be computationally expensive.
- Reduce the search space or use RandomizedSearchCV.
- Prioritize plausible settings and track compute budget.
- Validate the selected configuration on untouched data.

### Scenario 5: A forecasting model uses shuffled K-Fold
**Expected answer:**
- Random folds can train on future observations and validate on earlier observations.
- Use time-aware validation such as TimeSeriesSplit.
- Ensure feature generation also respects the prediction timestamp.

### Scenario 6: The best cross-validation model performs poorly on the final test set
**Expected answer:**
- Investigate overfitting to the validation process, distribution shift and test-set representativeness.
- Check whether preprocessing and feature generation were consistent.
- Review whether repeated model selection indirectly overused validation data.
- Report the final test result honestly.

## 5. Strong points to remember in interviews

- Cross-validation estimates performance across multiple splits.
- Keep a final untouched test set.
- Stratified K-Fold is useful for classification with imbalanced classes.
- Time series and grouped data need appropriate split strategies.
- GridSearchCV checks every specified combination.
- RandomizedSearchCV samples a fixed number of combinations.
- Fit preprocessing inside the cross-validation Pipeline.
- Large fold variation is useful information, not something to hide.
- Select models using validation evidence, not test-set feedback.

**Senior developer principle:** A good validation strategy must imitate how the model will encounter new data in the real application.
