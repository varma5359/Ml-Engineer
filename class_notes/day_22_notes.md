# Day 22: Multiple Linear Regression

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define multiple linear regression.
- Understand features, coefficients and intercept.
- Train a model with several input columns.
- Interpret coefficients carefully.
- Recognize multicollinearity and categorical encoding needs.

## 1. Definition
Multiple linear regression predicts one numeric target using two or more input features. It estimates a weighted sum of those features plus an intercept.

## 2. Syntax
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.coef_)
print(model.intercept_)
```

## 3. Three Business Examples
1. Predict house price from area, bedrooms and age.
2. Predict salary from experience, role level and education.
3. Predict monthly electricity usage from temperature, building size and previous usage.

## 4. Coefficient Interpretation
A coefficient represents the expected change in predicted target for a one-unit increase in that feature, holding the other included features constant. This is a model interpretation, not automatically a causal effect.

## 5. Scenario Questions
1. Area coefficient is positive. Does adding area always cause price to rise? Not necessarily; it is a conditional model association.
2. Coefficients change greatly after adding a related feature. What might be present? Multicollinearity.
3. Dataset includes city names. Can raw text be passed to LinearRegression? No; encode categorical features appropriately.
