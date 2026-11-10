# Day 25: Interview Questions — Logistic Regression and Sigmoid Function

## Beginner questions

### Q1. Define logistic regression and sigmoid function.
**Answer:** Logistic regression models class probability using a sigmoid transformation of a linear score. Despite its name, it is used for classification.

### Q2. Why would you use it?
**Answer:** It provides a systematic approach to learning patterns from data. Its suitability depends on the target, data quality, scale, assumptions and business cost of errors.

### Q3. Explain the basic syntax.
**Answer:** p = 1/(1+exp(-z)); LogisticRegression(max_iter=1000).fit(X_train,y_train); use predict_proba() for probabilities. I would explain the input, parameters, fitted state and output, not simply quote the class name.

## Practical questions

### Q4. How do you know the model is learning useful patterns rather than memorizing?
**Answer:** Compare training and validation performance, use an appropriate split, inspect error patterns and evaluate once on untouched test data. A large gap between training and validation can indicate overfitting.

### Q5. What would you inspect before tuning the model?
**Answer:** Target definition, missing values, data leakage, feature types, scaling needs, class balance, baseline performance and the validation strategy.

## Scenario questions

### Q6. A telecom company estimates churn probability. The probability can prioritize retention outreach, but a decision threshold must reflect campaign capacity and error costs.
**Answer:** I would clarify the business requirement, inspect examples and data quality, establish a baseline, train with leakage-safe preprocessing and validate on representative unseen data. I would compare relevant metrics, examine errors by segment and explain the cost of false positives and false negatives. I would not deploy based only on one aggregate score.

### Q7. Training performance is excellent but validation performance is poor. What might be happening?
**Answer:** The model may be overfitting, the split may not represent deployment, the dataset may be small, or there may be leakage or distribution differences. I would check split logic, pipeline boundaries, model complexity and segment coverage before changing hyperparameters.

### Q8. A stakeholder asks you to improve accuracy, but missing a positive case is very expensive. What do you recommend?
**Answer:** Clarify the cost of false negatives and false positives. Evaluate recall, precision, PR curve and confusion matrices across thresholds. Choose a threshold using validation data and business capacity, and explain the trade-off. Accuracy alone is not enough.

## Senior developer questions

### Q9. How do you decide whether this method is suitable for production?
**Answer:** Assess validated performance, latency, memory, explainability, stability, monitoring, data availability, retraining needs and operational ownership. A slightly weaker but reliable model may be preferable to a fragile complex one.

### Q10. What would you monitor after release?
**Answer:** Input schema and data quality, missing/unknown values, prediction distribution, latency, errors, drift and delayed ground-truth metrics. Review performance by relevant segment and define rollback criteria.

## Structured scenario-answer method
1. Clarify business impact.
2. Gather evidence and representative examples.
3. Identify likely causes without guessing.
4. Test hypotheses one at a time.
5. Choose a safe correction.
6. Validate on unseen and edge-case data.
7. Communicate trade-offs and monitor.
