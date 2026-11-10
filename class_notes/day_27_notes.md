# Day 27: K-Nearest Neighbors and Distance Metrics

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define KNN and how it makes predictions.
- Explain k and distance metrics.
- Understand why scaling matters.
- Recognize high-dimensionality and performance limits.

## 1. Definition
K-Nearest Neighbors (KNN) predicts using the labels or values of the nearest training observations under a chosen distance measure. It is called a lazy learner because it stores examples rather than learning a compact parameterized model during fitting.

## 2. Syntax
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 3. Three Examples
1. Classify a new shopper using similar customers' purchase frequency and spend.
2. Predict whether a flower belongs to a species using similar measurements.
3. Estimate a numeric property value from nearby examples (KNeighborsRegressor).

## 4. Important Choices
- k: number of neighbours.
- Distance: Euclidean/Manhattan or another supported metric.
- Scaling: important because features with large units can dominate distances.

## 5. Scenario Questions
1. Income dominates KNN distance compared with age. What should you do? Scale numeric features.
2. Why might k=1 overfit? It relies on one nearest observation and is sensitive to noise.
3. Prediction is slow on a huge dataset. What is the issue? KNN may require expensive neighbour searches at inference.
