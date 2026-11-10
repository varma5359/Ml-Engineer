# Day 24: Regularization: Ridge and Lasso

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define regularization and why it is used.
- Compare Ridge and Lasso.
- Understand alpha and coefficient shrinkage.
- Recognize why feature scaling and validation matter.

## 1. Definition
Regularization adds a penalty to the model objective to discourage excessively large coefficients. It can reduce variance and improve generalization.

- Ridge: L2 penalty; shrinks coefficients toward zero.
- Lasso: L1 penalty; can make some coefficients exactly zero.

## 2. Syntax
```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
```

## 3. Three Examples
1. Ridge when many correlated predictors should be retained but coefficient sizes need controlling.
2. Lasso when a sparse coefficient representation is useful.
3. Regularized regression when many campaign variables predict sales.

## 4. Alpha
Alpha controls penalty strength. Larger alpha generally means stronger regularization. Select it using validation or cross-validation, not the final test set. Scale features before regularized regression when feature units differ.

## 5. Scenario Questions
1. Coefficients are extremely large and unstable. What can you try? Scaling and regularization.
2. Lasso sets a coefficient to zero. What does that mean? That feature's coefficient was removed by the fitted penalized model, not necessarily that the feature is universally useless.
3. Should alpha be chosen from test performance? No; use validation or cross-validation.
