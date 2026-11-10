# Day 83: Interview Questions — Capstone Tuning and Validation

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. What is hyperparameter tuning?
**Answer:**
- Searching model configuration settings.
- Examples include tree depth, number of trees and regularization strength.
- It uses validation or cross-validation evidence.
- It must be controlled to avoid overfitting the validation process.

**Follow-up:** What is the difference between a parameter and a hyperparameter?

### Q2. Explain GridSearchCV.
**Answer:**
- It evaluates every combination in a defined parameter grid.
- It uses cross-validation to estimate each configuration's score.
- It selects the best configuration according to the chosen metric.
- It refits the selected estimator when configured to do so.

**Follow-up:** What is the computational cost of a 20-configuration grid with five folds?

### Q3. What is StratifiedKFold?
**Answer:**
- A cross-validation splitter for classification.
- It aims to preserve class proportions across folds.
- It helps when classes are imbalanced.
- It is not a substitute for time-aware or group-aware splitting when dependencies exist.

**Follow-up:** Why can random stratification be inappropriate for monthly customer snapshots?

### Q4. Why include preprocessing inside the Pipeline during tuning?
**Answer:**
- Each fold must learn transformations only from its training data.
- It prevents information from validation folds influencing imputers or encoders.
- It keeps training and inference transformations consistent.
- It makes the workflow easier to reproduce.

**Follow-up:** What kind of leakage can target encoding introduce?

### Q5. How do you select a scoring metric for tuning?
**Answer:**
- Start from the business objective and class balance.
- Choose a metric such as average precision, recall or F1.
- Consider threshold-dependent and threshold-independent requirements.
- Document why the metric is appropriate.

**Follow-up:** Why might accuracy be unsuitable for a rare churn class?

## 2. Scenario questions

### Q6. Grid search takes too long. What can you do?
**Answer:**
- Reduce the search space to justified values.
- Use RandomizedSearchCV.
- Use fewer folds during early exploration, then validate carefully.
- Limit parallel jobs based on available resources.
- Consider simpler candidates and early stopping where supported.

**Follow-up:** What trade-off comes with reducing the number of folds?

### Q7. Best CV score is 0.842 and second best is 0.841. Which do you choose?
**Answer:**
- Compare variation across folds.
- Consider model complexity and runtime.
- Check whether the difference is practically meaningful.
- Prefer a simpler model if performance is effectively equivalent and requirements support it.

**Follow-up:** How could repeated cross-validation help assess stability?

### Q8. A tuned model performs worse on the final test set. What do you investigate?
**Answer:**
- Test distribution shift and sample size.
- Validation overfitting from repeated search.
- Data leakage or split design.
- Randomness and reproducibility.
- Whether the test set represents the intended future population.

**Follow-up:** How would you report this honestly to stakeholders?

## 3. Project questions

### Q9. Describe your Optimized model deliverable.
**Answer:**
- I selected a candidate from validation comparison.
- I defined a bounded hyperparameter search.
- I used cross-validation with preprocessing inside the pipeline.
- I recorded the best configuration and fold variation.
- I preserved the final test set for the final evaluation.

**Follow-up:** What metadata would you log in MLflow?

### Q10. Why is the best cross-validation score not a guarantee of production performance?
**Answer:**
- Cross-validation estimates performance on sampled folds.
- Production data can differ.
- The metric may not capture all business and operational requirements.
- Monitoring and post-deployment validation remain necessary.

**Follow-up:** What signals would trigger a model review?

## 4. Rapid revision
- Hyperparameter: estimator configuration.
- GridSearchCV: exhaustive specified search.
- RandomizedSearchCV: sampled search.
- StratifiedKFold: preserves class proportions approximately.
- Pipeline: prevents fold-level preprocessing leakage.
- Final test: reserved for final assessment.
