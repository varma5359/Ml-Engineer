# Day 16: Interview Questions — Data Leakage and Preprocessing Pipelines

## Beginner questions

### Q1. What is data leakage and preprocessing pipelines?
**Answer:** Data leakage happens when information unavailable at prediction time, or information from validation/test data, influences model training or evaluation.

### Q2. Why is it important in an ML workflow?
**Answer:** It improves the reliability of data preparation, training or evaluation. Leakage creates falsely optimistic metrics and production failure. Audit when every feature becomes available and split data before fitting transformations.

### Q3. What syntax or method would you use?
**Answer:** Fit imputation, scaling, encoding and feature selection only on training folds. Use sklearn Pipeline and ColumnTransformer so transformations are fitted and applied consistently. I would explain the input, parameters, output and validation rather than reciting a function name.

## Practical questions

### Q4. How do you know the operation worked correctly?
**Answer:** Test a small known example, compare expected and actual output, inspect boundary cases and verify that the result still makes sense in the business domain.

### Q5. What beginner mistake should you avoid?
**Answer:** Applying a technique without checking assumptions or the meaning of the data. I would inspect source definitions, data types, units, missingness and output before using it downstream.

## Scenario-based interview questions

### Q6. A hospital model predicts readmission at discharge but uses a field entered after readmission. Offline metrics look excellent, but the feature cannot exist when a real prediction is made.
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


---

## Detailed Scenario and Technical Interview Expansion

### Scenario: hospital readmission
A model predicts readmission at discharge but includes a field updated only after readmission. The offline score is excellent but the field is unavailable at prediction time.

**Senior answer:** Remove or redefine the feature, confirm timestamps with the data owner, rebuild the pipeline and evaluate using a split that reflects deployment. Investigate suspiciously high metrics before celebrating.

**Q1. Why is scaling before train-test split leakage?** It learns test-set distribution statistics.  
**Q2. Is customer ID always leakage?** Not automatically, but it can cause memorization and poor generalization; evaluate based on the intended unseen-entity scenario.  
**Q3. How does Pipeline help?** It keeps learned transformations inside each training fold during cross-validation and applies them consistently at inference.  
**Q4. A feature has a strong correlation with the target. Is it safe?** Check when and how it is created; strong correlation may indicate leakage.  
**Q5. What if the test score is 99.9% for a difficult task?** Audit labels, split logic, duplicates, timestamps and feature availability before assuming exceptional performance.
