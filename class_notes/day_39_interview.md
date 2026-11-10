# Day 39: Interview Questions — Association Rules and Market Basket Analysis

## Beginner concepts

### Q1. Explain association rules and market basket analysis.
**Answer:** Association rule mining finds items that frequently occur together in transactions. A rule A → B describes how often B appears when A appears; it does not prove A causes B.

### Q2. Why is this method useful?
**Answer:** It addresses a practical pattern-discovery or prediction need. Its suitability depends on the dataset, assumptions, target/goal and operational constraints.

### Q3. Explain the basic implementation.
**Answer:**
```python
Support(A→B) = transactions containing A and B / all transactions. Confidence = support(A and B) / support(A). Lift = confidence(A→B) / support(B). Use mlxtend.frequent_patterns.apriori and association_rules after one-hot transaction encoding.
```
I would explain the inputs, parameters, what fit learns, what the output represents and how I validate it.

## Practical and scenario questions

### Q4. A grocery retailer finds customers who buy pasta often also buy pasta sauce. It may test bundle placement, but must check margin, stock and whether the association is commercially useful.
**Answer:** I would confirm the business objective and data meaning, inspect representative records, prepare features according to the algorithm's requirements, establish a baseline, fit using the appropriate training data and evaluate the output. I would inspect errors or group profiles, document assumptions and verify that the result is actionable.

### Q5. What is the most important limitation to explain to a stakeholder?
**Answer:** High confidence can be misleading when B is already very popular. Lift, support, time windows and business constraints matter; association is not causation.

### Q6. The result changes significantly when you change one parameter. What do you do?
**Answer:** Run a controlled sensitivity analysis across a reasonable parameter range, keep the data and evaluation method fixed, measure quality and stability, and choose based on validation and business usefulness rather than a single attractive result.

### Q7. A colleague says the algorithm's output is automatically the correct business answer. Do you agree?
**Answer:** No. Algorithms produce outputs under assumptions. A developer must check data quality, interpret the result, compare with domain knowledge and validate against business outcomes.

### Q8. How would you prepare this method for production?
**Answer:** Package preprocessing and the estimator consistently, validate input schema and ranges, record model/data versions, test edge cases, monitor input drift and outcomes, and define an owner and rollback procedure.

## Senior developer questions

### Q9. How do you compare this method with an alternative?
**Answer:** Use the same data protocol and compare relevant quality measures, stability, interpretability, latency, memory, operational complexity and business impact. Avoid choosing solely by algorithm popularity.

### Q10. What would you monitor after deployment?
**Answer:** Input quality, feature distributions, prediction/cluster distribution, runtime failures, latency, delayed labels or business outcomes, and changes across important segments. Define alerts and review ownership.

## Interview answer structure
Clarify the business question → explain the concept → state assumptions → describe implementation → validate output → explain limitations → recommend monitoring.
