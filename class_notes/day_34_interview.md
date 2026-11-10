# Day 34: Interview Questions — K-Means and Centroid-Based Clustering

## Beginner concepts

### Q1. Explain k-means and centroid-based clustering.
**Answer:** K-Means partitions observations into K clusters by assigning each point to the nearest centroid and repeatedly updating centroids to reduce within-cluster squared distances.

### Q2. Why is this method useful?
**Answer:** It addresses a practical pattern-discovery or prediction need. Its suitability depends on the dataset, assumptions, target/goal and operational constraints.

### Q3. Explain the basic implementation.
**Answer:**
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels = model.fit_predict(X_scaled)
centres = model.cluster_centers_
```
I would explain the inputs, parameters, what fit learns, what the output represents and how I validate it.

## Practical and scenario questions

### Q4. A subscription business groups customers using monthly usage and support contacts. Centroids describe the typical profile of each group in scaled feature space.
**Answer:** I would confirm the business objective and data meaning, inspect representative records, prepare features according to the algorithm's requirements, establish a baseline, fit using the appropriate training data and evaluate the output. I would inspect errors or group profiles, document assumptions and verify that the result is actionable.

### Q5. What is the most important limitation to explain to a stakeholder?
**Answer:** K must be selected, numeric features need meaningful scaling, and K-Means is sensitive to outliers and non-spherical cluster shapes. Different initializations can produce different results.

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
