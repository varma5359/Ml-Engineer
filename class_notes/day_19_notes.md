# Day 19: Linear Regression, Assumptions and Equation

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define regression and linear regression.
- Understand the equation, coefficients and intercept.
- Train a simple linear regression model.
- Interpret residuals and basic assumptions.
- Recognize when a linear model may be unsuitable.

## 1. What Is Regression?
Regression predicts a numeric quantity, such as house price, delivery duration or monthly demand.

## 2. What Is Linear Regression?
Linear regression estimates a numeric target as a weighted sum of input features plus an intercept.

Equation in words: predicted value = intercept + (coefficient × feature) for each feature.

- Intercept: starting value when features are zero.
- Coefficient: expected target change for a one-unit feature increase, holding other included features constant.
- Feature: input value.
- Prediction: model's estimated target.

## 3. Simple Linear Regression Syntax
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.coef_)
print(model.intercept_)
```

**Three examples**
1. Predict house price from area.
2. Predict delivery time from distance.
3. Predict monthly electricity usage from historical usage and weather features.

## 4. Multiple Linear Regression
With multiple features, the model combines several weighted inputs.

```python
model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)
```
Here X_train can contain multiple feature columns.

## 5. Assumptions and Diagnostics
Check whether the chosen feature representation supports a reasonably linear relationship, inspect residual patterns, investigate influential points and consider independence of observations. Normal residuals are important for some classical inference procedures, not a universal requirement for prediction.

## 6. Scenario Questions
1. House-price errors are systematically higher for premium properties. What do you inspect? Residuals by price band and missing features.
2. A coefficient is positive. Does it prove the feature causes the target to rise? No.
3. Model predicts well on training but poorly on test. What may be happening? Overfitting, distribution mismatch or leakage/split issues.
