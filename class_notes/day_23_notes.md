# Day 23: Polynomial Regression and Nonlinear Relationships

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain why a straight-line model may be insufficient.
- Define polynomial features.
- Generate squared and interaction terms.
- Understand model degree and overfitting.

## 1. Definition
Polynomial regression creates transformed features such as x² or x³ and fits a linear regression model using them. The model remains linear in its learned coefficients but can represent a curved relationship in the original feature.

## 2. Syntax
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 3. Three Examples
1. Advertising spend and sales with diminishing returns.
2. Temperature and energy use with a curved relationship.
3. Vehicle speed and fuel consumption across different speed ranges.

## 4. Degree
Degree 2 includes squared terms; degree 3 includes cubic terms. Higher degree adds flexibility but can fit noise and become unstable outside the observed data range.

## 5. Scenario Questions
1. A linear model misses a curve in residuals. What could you test? Polynomial features or another nonlinear model.
2. A degree-12 model fits training perfectly but fails validation. What is likely? Overfitting.
3. Is a polynomial model safe for far-future extrapolation? Often not; validate the range and use domain knowledge.
