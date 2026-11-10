# Day 20: Cost Functions and Residual Analysis

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define residual and loss function.
- Calculate and interpret MAE, MSE and RMSE.
- Understand R-squared.
- Inspect error patterns rather than relying on one metric.
- Select metrics based on business impact.

## 1. What Is a Residual?
A residual is actual value minus predicted value for one observation.

- Positive residual: prediction was too low.
- Negative residual: prediction was too high.
- Zero residual: exact prediction.

## 2. What Is a Loss Function?
A loss function measures how wrong a prediction is. Training algorithms use a loss/objective to find model parameters that reduce error.

## 3. Common Regression Metrics
- MAE: average absolute error; easy to interpret in target units.
- MSE: average squared error; gives large errors greater influence.
- RMSE: square root of MSE; returns to target units.
- R-squared: compares model performance with a mean-target reference; can be negative on test data.

### Syntax
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print(mae, mse, rmse, r2)
```

## 4. Three Interpretation Examples
1. MAE = 5 minutes for delivery prediction means the average absolute error is 5 minutes.
2. RMSE much larger than MAE may indicate some large errors.
3. Negative test R² means the model is worse than predicting the test-set mean under the R² comparison.

## 5. Residual Analysis
Inspect residuals against predictions, time and business segments. A curved pattern may indicate nonlinearity; a widening spread may indicate changing error variance; one segment with large residuals may need investigation.

## 6. Scenario Questions
1. A model has low average error but consistently underpredicts luxury houses. What do you do? Inspect residuals by price band.
2. Why can RMSE be much larger than MAE? Squaring gives large errors more weight.
3. Can R² be negative? Yes, on evaluation data.
