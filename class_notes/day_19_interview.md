# Day 19: Interview Questions — Linear Regression, Assumptions and Equation

## Beginner questions

### Q1. What is linear regression, assumptions and equation?
**Answer:** Linear regression estimates a continuous target as a weighted sum of input features plus an intercept.

### Q2. Why is it important in an ML workflow?
**Answer:** It improves the reliability of data preparation, training or evaluation. Check residual patterns, non-linearity, influential points and data quality. Normal residuals are relevant to some statistical inference, not a universal requirement for prediction.

### Q3. What syntax or method would you use?
**Answer:** y_hat = b0 + b1*x1 + ... + bp*xp. In scikit-learn use LinearRegression().fit(X_train, y_train), then predict(X_test). Inspect coefficients and residuals. I would explain the input, parameters, output and validation rather than reciting a function name.

## Practical questions

### Q4. How do you know the operation worked correctly?
**Answer:** Test a small known example, compare expected and actual output, inspect boundary cases and verify that the result still makes sense in the business domain.

### Q5. What beginner mistake should you avoid?
**Answer:** Applying a technique without checking assumptions or the meaning of the data. I would inspect source definitions, data types, units, missingness and output before using it downstream.

## Scenario-based interview questions

### Q6. A property company estimates sale price using area, bedrooms and age. The model is an estimate, not a guaranteed market valuation; location and market changes can matter.
**Answer:** I would first clarify the business goal and gather examples of the issue. Then I would inspect source data, assumptions and the point in time at which each field becomes available. I would reproduce the problem on a small sample, test likely causes one at a time, select a correction based on evidence and validate it on held-out data. I would document limitations and monitor the outcome after release.

### Q7. The aggregate metric is acceptable, but one important customer segment performs poorly. What do you do?
**Answer:** I would calculate metrics and inspect data quality by segment, verify sample sizes, investigate whether the segment differs in behaviour or data capture, and discuss the error trade-off with the business owner. I would not hide the issue behind an overall average.

### Q8. A teammate suggests an immediate fix without investigating the cause. How do you respond?
**Answer:** I would ask for evidence and reproduce the failure. A quick change can mask the root cause or create a new defect. I would compare hypotheses, make the smallest safe change and add a regression test.

## Senior developer questions

### Q9. How do you choose between a simple and complex solution?
**Answer:** Compare correctness, maintainability, latency, cost, explainability, data requirements and operational risk. Start with a baseline and add complexity only when repeatable validation and business impact justify it.

### Q10. What would you monitor after deployment?
**Answer:** Input quality, missing and unknown values, output distributions, latency, errors, delayed ground-truth metrics and segment-level outcomes. Define owners and alert thresholds before release.

## Answer framework
1. Clarify the decision and constraints.
2. Inspect the data and assumptions.
3. Diagnose using evidence.
4. Choose the smallest safe correction.
5. Validate normal and edge cases.
6. Communicate trade-offs and limitations.
7. Monitor after release.
