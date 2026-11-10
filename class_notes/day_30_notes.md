# Day 30: Random Forests and Ensemble Learning

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define ensemble learning and random forest.
- Explain bootstrap samples and random feature selection.
- Train a random forest classifier/regressor.
- Compare a forest with a single decision tree.
- Understand limitations and feature importance.

## 1. What Is Ensemble Learning?
Ensemble learning combines predictions from multiple models to produce a final prediction. The goal is often to improve stability or predictive performance.

## 2. What Is a Random Forest?
A random forest trains many decision trees using bootstrap samples and random subsets of features. Classification combines votes; regression averages predictions.

## 3. Syntax
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 4. Three Examples
1. Predict insurance claim risk from customer and policy details.
2. Predict customer churn using usage and support data.
3. Classify product quality using sensor measurements.

## 5. Important Parameters
- n_estimators: number of trees.
- max_features: number/subset of features considered at a split.
- max_depth: maximum tree depth.
- min_samples_leaf: minimum observations required in a leaf.
- random_state: controls randomness for reproducibility.
- n_jobs: parallel jobs.

## 6. Scenario Questions
1. A single tree changes greatly after adding a few records. Why might a forest help? Averaging diverse trees can reduce variance.
2. Forest performs well but is hard to explain. What can you provide? Segment metrics, feature diagnostics and appropriate explanation tools.
3. Can a random forest extrapolate far beyond training target ranges? Usually poorly in regression; predictions are based on averages of training outcomes.
