# Day 18: ML Workflow, Baselines and Reproducibility

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Describe a complete basic ML workflow.
- Define a baseline and explain why it matters.
- Use DummyClassifier or DummyRegressor.
- Record experiment settings for reproducibility.
- Compare candidate models fairly.

## 1. What Is an ML Workflow?
A repeatable sequence of steps used to build, evaluate and operate a model.

Problem definition → data understanding → split → preprocessing → baseline → model training → evaluation → deployment → monitoring.

## 2. What Is a Baseline?
A baseline is a simple reference prediction or existing business rule. A complex model should demonstrate useful improvement over it.

**Example 1 — Classification majority baseline**
```python
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
pred = baseline.predict(X_test)
```

**Example 2 — Regression mean baseline**
```python
from sklearn.dummy import DummyRegressor
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
pred = baseline.predict(X_test)
```

**Example 3 — Business-rule baseline**
A delivery team predicts all deliveries using the historical median delivery time. A model should improve on this reference under a fair evaluation.

## 3. Reproducibility
Record dataset version, code revision, split method, random seed, preprocessing, library versions, model parameters and metrics.

## 4. Scenario Questions
1. Complex model accuracy is lower than the majority-class baseline. What do you do? Investigate relevant metrics and whether the complex model adds business value.
2. A colleague cannot reproduce your score. What records help? Dataset version, code, split, random seed, package versions and parameters.
3. Model worked last month but fails now. What do you check? Data drift, schema changes, feature availability and operational pipeline.
