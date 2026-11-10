# Day 31: Interview Questions — Support Vector Machines and Maximum-Margin Hyperplanes

## Beginner concepts

### Q1. Explain support vector machines and maximum-margin hyperplanes.
**Answer:** SVM is a supervised algorithm that finds a decision boundary separating classes while maximizing the margin between the closest training points (support vectors).

### Q2. Why is this method useful?
**Answer:** It addresses a practical pattern-discovery or prediction need. Its suitability depends on the dataset, assumptions, target/goal and operational constraints.

### Q3. Explain the basic implementation.
**Answer:**
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
model.fit(X_train, y_train)
pred = model.predict(X_test)
```
I would explain the inputs, parameters, what fit learns, what the output represents and how I validate it.

## Practical and scenario questions

### Q4. A bank separates legitimate and suspicious transactions using several behavioural features. SVM can form a boundary that is not a simple single-feature threshold.
**Answer:** I would confirm the business objective and data meaning, inspect representative records, prepare features according to the algorithm's requirements, establish a baseline, fit using the appropriate training data and evaluate the output. I would inspect errors or group profiles, document assumptions and verify that the result is actionable.

### Q5. What is the most important limitation to explain to a stakeholder?
**Answer:** SVM is sensitive to feature scaling and can be expensive on large datasets. A complex kernel can overfit. Tune C and kernel parameters using validation data.

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
