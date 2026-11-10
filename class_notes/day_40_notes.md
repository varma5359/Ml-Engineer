# Day 40: Anomaly Detection Concepts

**Pathway:** Machine Learning Engineer  
**Phase:** 4 — Unsupervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define anomaly detection.
- Distinguish supervised, unsupervised and semi-supervised anomaly detection.
- Understand Isolation Forest at a high level.
- Interpret anomaly flags as investigation signals.
- Recognize false positives and false negatives.

## 1. What Is Anomaly Detection?
Anomaly detection identifies observations that differ substantially from expected patterns. An anomaly is unusual; it is not automatically fraud, failure or an error.

## 2. Approaches
- Supervised: train with labelled normal and anomaly examples.
- Unsupervised: identify unusual patterns without complete labels.
- Semi-supervised/one-class: learn the normal pattern and flag deviations.

## 3. Isolation Forest Example
Isolation Forest isolates observations by random feature splits. Unusual points can often be isolated with fewer splits.

```python
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_train)
labels = model.predict(X_test)
```
In scikit-learn, -1 indicates an outlier and 1 indicates an inlier. contamination is an assumption about the expected outlier proportion used by the estimator.

## 4. Three Applications
1. Unusual payment patterns.
2. Unexpected machine sensor readings.
3. Abnormal network activity.

## 5. Scenario Questions
1. A legitimate premium customer is flagged. What kind of error? False positive.
2. Fraudulent activity is not flagged. What kind of error? False negative.
3. Should every anomaly automatically block a transaction? Not necessarily; design review, risk thresholds and customer impact controls.
