# Day 81: Interview Questions — Capstone Candidate Model Training

**Phase:** 8 — End-to-End ML Capstone

## 1. Core questions

### Q1. Why train candidate models?
**Answer:**
- Different algorithms learn different patterns.
- One algorithm is not best for every dataset.
- Comparable experiments provide evidence for selection.
- Model complexity and operational constraints also matter.

**Follow-up:** How many candidate models are enough for a small capstone?

### Q2. Why use the same split for every candidate?
**Answer:**
- It controls variation caused by different samples.
- It makes model comparisons more meaningful.
- It supports reproducible experiment records.
- The split must reflect the intended deployment scenario.

**Follow-up:** When would a time-based split be more appropriate?

### Q3. Compare Logistic Regression and Random Forest.
**Answer:**
- Logistic Regression models a linear decision relationship in transformed feature space.
- Random Forest combines decision trees and can capture nonlinear interactions.
- Logistic Regression is often easier to explain and lighter to serve.
- Random Forest may model more complex relationships but can be larger.

**Follow-up:** How would you explain a Random Forest prediction?

### Q4. What is class_weight="balanced"?
**Answer:**
- It adjusts class weights inversely based on class frequencies.
- It can increase the influence of minority-class errors.
- It does not guarantee better recall or precision.
- It must be validated for the business objective.

**Follow-up:** Does class weighting change the class distribution in the dataset?

### Q5. What should be recorded for a training run?
**Answer:**
- Dataset and code version.
- Split method and random seed.
- Feature and target definition.
- Model and hyperparameters.
- Metrics, runtime and artifact reference.
- Software environment and limitations.

**Follow-up:** How would MLflow support this record?

## 2. Scenario questions

### Q6. A tree model scores better, but training takes much longer. What do you consider?
**Answer:**
- Whether the metric gain is meaningful and stable.
- Inference latency and memory.
- Training frequency and compute cost.
- Interpretability and operational support.
- Whether the gain improves business outcomes.

**Follow-up:** What if retraining happens every hour?

### Q7. One candidate cannot handle sparse one-hot encoded data. What do you do?
**Answer:**
- Check the estimator's input requirements.
- Consider a compatible estimator or sparse-aware preprocessing.
- Avoid converting high-dimensional data to dense form without checking memory.
- Document the reason for the chosen design.

**Follow-up:** How can high-cardinality categorical features affect memory?

### Q8. Candidate metrics vary significantly across folds. What does that suggest?
**Answer:**
- Performance may be sensitive to the sample.
- The dataset may be small or heterogeneous.
- Review split strategy, segments and data quality.
- Report variation rather than only the mean score.

**Follow-up:** How would you communicate uncertainty to a stakeholder?

## 3. Project questions

### Q9. Describe your Model training suite.
**Answer:**
- I reused a common preprocessing pipeline.
- I trained multiple candidate classifiers.
- I evaluated them on a consistent validation strategy.
- I recorded hyperparameters, metrics and runtime.
- I selected a provisional candidate for deeper evaluation.

**Follow-up:** Why is it only a provisional selection at this stage?

### Q10. How do you avoid overfitting during model comparison?
**Answer:**
- Use training data for fitting.
- Use validation or cross-validation for selection.
- Keep the final test set untouched.
- Limit repeated manual experimentation against the same validation set and report uncertainty.

**Follow-up:** What role does nested cross-validation play?

## 4. Rapid revision
- Candidate: model considered for selection.
- Validation: data used for model decisions.
- Class weighting: changes class error emphasis.
- PR-AUC: positive-class ranking metric.
- Reproducibility: data, code, config and environment records.
- Final test: reserved for final estimate.
