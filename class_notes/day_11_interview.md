# Day 11: Interview Questions — Outlier Detection and Treatment

## Beginner questions

### Q1. What is outlier detection and treatment?
**Answer:** An outlier is a value unusually far from the pattern of a dataset. It may be a data-entry error, a rare but valid event, or evidence of a new business condition.

### Q2. Why is it important in an ML workflow?
**Answer:** It improves the reliability of data preparation, training or evaluation. Blind deletion can remove the exact fraud cases or premium customers the model must detect. Compare model performance with and without treatment, using thresholds learned from training data only.

### Q3. What syntax or method would you use?
**Answer:** Use domain checks, box plots, IQR fences and z-scores as investigation tools. IQR = Q3 - Q1; lower fence = Q1 - 1.5 × IQR; upper fence = Q3 + 1.5 × IQR. I would explain the input, parameters, output and validation rather than reciting a function name.

## Practical questions

### Q4. How do you know the operation worked correctly?
**Answer:** Test a small known example, compare expected and actual output, inspect boundary cases and verify that the result still makes sense in the business domain.

### Q5. What beginner mistake should you avoid?
**Answer:** Applying a technique without checking assumptions or the meaning of the data. I would inspect source definitions, data types, units, missingness and output before using it downstream.

## Scenario-based interview questions

### Q6. A ₹9 lakh payment among mostly ₹500–₹10,000 transactions might be fraud, a genuine bulk payment or an extra zero. Verify transaction source and customer context before deleting or capping it.
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

### Scenario: insurance claim values suddenly increase
An insurer sees several claims much larger than the usual claim amount. A junior developer removes every value above the IQR fence. This improves the training score, but the model now performs poorly on expensive claims.

**Senior engineer response:** First verify whether the claims are duplicates, currency/unit errors, genuine catastrophic claims or a change in the portfolio. Check source records and related fields, not only the amount column. If they are valid, retain them or use a robust modelling strategy. Evaluate performance separately for ordinary and high-value claims. Record the decision and its business owner.

### Scenario questions
**Q1. A value is flagged by IQR. Does that prove it is incorrect?**  
No. IQR is a statistical screening rule, not a business truth. Verify the record against source data, units, domain limits and related features.

**Q2. Should outlier thresholds be calculated before train-test splitting?**  
No, when the thresholds influence model preprocessing. Split first and estimate thresholds on training data, then apply the learned rule to validation/test data.

**Q3. A fraud model's rare high-value transactions are removed as outliers. What is the risk?**  
The cleaning step may remove the very events the model is intended to detect. Rare does not mean invalid.

**Q4. When might you prefer a log transformation over deletion?**  
When values are valid and positive but strongly right-skewed, and reducing scale differences helps the model. Interpret transformed effects carefully.

**Q5. How would you prove an outlier treatment helped?**  
Compare a baseline and treated approach using the same split, relevant metrics, segment-level errors and business costs. Keep a final test set untouched.
