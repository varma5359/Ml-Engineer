# Day 47: Regression Evaluation — MAE, MSE, RMSE and R²

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives
- Explain why regression predictions need error evaluation.
- Distinguish actual values, predicted values and residuals.
- Understand MAE, MSE, RMSE and R² in business language.
- Calculate regression metrics using scikit-learn.
- Compare metrics and identify their limitations.
- Interpret errors in original business units.
- Select metrics based on the impact of prediction errors.

## 2. What is regression evaluation?

Regression predicts a numerical value, such as:
- House selling price.
- Monthly sales.
- Delivery time.
- Energy consumption.
- Product demand.
- Customer lifetime value.

A prediction is rarely exactly equal to the actual value. Regression evaluation describes how far predictions are from observed outcomes and whether the model is useful for the intended decision.

Example:
- Actual house price: ₹50 lakh.
- Predicted house price: ₹47 lakh.
- Prediction error: ₹3 lakh below the actual value.

A useful evaluation should tell us:
- How large are the typical errors?
- Are some errors unusually large?
- Does the model perform better than a simple baseline?
- Does it work across different price ranges or customer groups?
- Are the errors acceptable for the business?

## 3. Actual values, predicted values and residuals

- **Actual value:** Known target value in the evaluation dataset.
- **Predicted value:** Value returned by the model.
- **Residual:** Actual value minus predicted value.
- **Prediction error:** Difference between actual and predicted values; some reports use absolute error to ignore direction.

Residual interpretation:
- Positive residual: model predicted below the actual value.
- Negative residual: model predicted above the actual value.
- Residual near zero: prediction is close to actual value.

## 4. Create a simple regression example

### 4.1 Import libraries

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Libraries ready")
~~~

**Expected output**
~~~text
Libraries ready
~~~

**Explanation**
NumPy and Pandas prepare values. Matplotlib helps visualize actual values and predictions.

### 4.2 Create actual target values

~~~python
y_actual = np.array([100, 150, 200, 250, 300, 350])

print(y_actual)
~~~

**Expected output**
~~~text
[100 150 200 250 300 350]
~~~

**Explanation**
These values could represent actual monthly sales in thousands of rupees for six observations.

### 4.3 Create predicted values

~~~python
y_predicted = np.array([110, 140, 190, 270, 280, 360])

print(y_predicted)
~~~

**Expected output**
~~~text
[110 140 190 270 280 360]
~~~

**Explanation**
These are the values predicted by a regression model for the same six observations.

### 4.4 Create a comparison table

~~~python
comparison = pd.DataFrame({
    "actual": y_actual,
    "predicted": y_predicted
})

print(comparison)
~~~

**Expected output**
A six-row table with actual and predicted values.

**Explanation**
The actual and predicted values must be aligned row by row. If their order is mismatched, all regression metrics become meaningless.

### 4.5 Calculate residuals

~~~python
comparison["residual"] = (
    comparison["actual"] - comparison["predicted"]
)

print(comparison)
~~~

**Expected output**
A table with a residual column containing positive and negative values.

**Explanation**
A positive residual means the model underpredicted. A negative residual means it overpredicted.

### 4.6 Calculate absolute errors

~~~python
comparison["absolute_error"] = comparison["residual"].abs()

print(comparison)
~~~

**Expected output**
The same table with non-negative absolute error values.

**Explanation**
Absolute error measures the size of the mistake without considering whether the model predicted too high or too low.

## 5. Mean Absolute Error (MAE)

**MAE** is the average size of prediction errors when direction is ignored.

Business interpretation:
- If MAE is 12,000 rupees, the average absolute prediction error is about ₹12,000 in the evaluation sample.
- MAE is easy to explain because it uses the same unit as the target.
- Every error contributes according to its absolute size.

Use MAE when:
- You want an easy-to-explain average error.
- Large errors should matter, but you do not want them amplified as strongly as squared-error metrics.
- The target's original unit is meaningful to stakeholders.

Limitations:
- MAE does not show whether the model overpredicts or underpredicts.
- It can hide a small number of very large errors in the average.
- Compare it with residual analysis and business requirements.

### 5.1 Import mean_absolute_error

~~~python
from sklearn.metrics import mean_absolute_error

print("MAE function imported")
~~~

**Expected output**
~~~text
MAE function imported
~~~

### 5.2 Calculate MAE

~~~python
mae = mean_absolute_error(y_actual, y_predicted)

print("MAE:", mae)
~~~

**Expected output**
A numeric value representing the average absolute error in the target's units.

**Explanation**
The metric compares each actual value with its corresponding prediction and averages the absolute differences.

### 5.3 Calculate MAE manually

~~~python
absolute_errors = np.abs(y_actual - y_predicted)
manual_mae = absolute_errors.mean()

print("Absolute errors:", absolute_errors)
print("Manual MAE:", manual_mae)
~~~

**Expected output**
An array of absolute errors and a single average error.

**Explanation**
The manual version helps understand the metric. Use the library implementation in project code and verify input alignment.

## 6. Mean Squared Error (MSE)

**MSE** averages squared prediction errors.

Business interpretation:
- Larger errors receive much more influence because the errors are squared.
- MSE is useful when large mistakes should be penalized heavily.
- Its unit is the square of the target unit, so it is less intuitive to explain directly.

Use MSE when:
- Large prediction mistakes are especially costly.
- You are comparing models on the same target and evaluation data.
- The modelling method or optimization objective uses squared error.

Limitations:
- Sensitive to large outliers.
- Not expressed in the original target unit.
- A high MSE does not tell you which observations caused the problem.

### 6.1 Import mean_squared_error

~~~python
from sklearn.metrics import mean_squared_error

print("MSE function imported")
~~~

### 6.2 Calculate MSE

~~~python
mse = mean_squared_error(y_actual, y_predicted)

print("MSE:", mse)
~~~

**Expected output**
A numeric squared-error value.

**Explanation**
Each prediction error is squared before averaging, so a large error contributes more heavily.

### 6.3 Compare absolute error and squared error

~~~python
error_table = pd.DataFrame({
    "absolute_error": absolute_errors,
    "squared_error": (y_actual - y_predicted) ** 2
})

print(error_table)
~~~

**Expected output**
A table showing absolute and squared errors for each observation.

**Explanation**
Notice how squared error grows more rapidly for larger mistakes.

## 7. Root Mean Squared Error (RMSE)

**RMSE** is the square root of MSE.

Business interpretation:
- RMSE is expressed in the same unit as the target.
- Large errors receive more influence than they do under MAE.
- It is often easier to explain than MSE.

Use RMSE when:
- Large prediction errors deserve extra penalty.
- You want an error measure in the target's original unit.
- You are comparing models on the same dataset and target scale.

Limitations:
- Sensitive to large errors.
- Does not reveal whether errors are overpredictions or underpredictions.
- It should be considered with residual plots and MAE.

### 7.1 Calculate RMSE

~~~python
rmse = np.sqrt(mse)

print("RMSE:", rmse)
~~~

**Expected output**
A numeric value in the original target unit.

**Explanation**
RMSE converts the squared-error summary back into the target's unit.

### 7.2 Use scikit-learn root mean squared error where available

~~~python
try:
    from sklearn.metrics import root_mean_squared_error

    rmse_library = root_mean_squared_error(
        y_actual,
        y_predicted
    )
except ImportError:
    rmse_library = np.sqrt(
        mean_squared_error(y_actual, y_predicted)
    )

print("RMSE:", rmse_library)
~~~

**Expected output**
The same RMSE value as the square root of MSE, subject to numeric precision.

**Explanation**
The dedicated function is available in newer scikit-learn versions. If an older environment does not provide it, use the square root of mean_squared_error.

## 8. R² score

**R²**, often called the coefficient of determination, compares the model with a simple baseline that predicts the average target value.

Practical interpretation:
- R² near 1: predictions explain much of the variation in the evaluation target.
- R² near 0: performance is similar to predicting the target mean under the usual evaluation comparison.
- R² below 0: model predictions are worse than that mean baseline on the evaluation data.

Important:
- R² is not the percentage of predictions that are correct.
- R² can be negative on test data.
- R² does not tell us the size of errors in business units.
- A high R² does not guarantee acceptable prediction errors for every segment.

### 8.1 Import r2_score

~~~python
from sklearn.metrics import r2_score

print("R2 function imported")
~~~

### 8.2 Calculate R²

~~~python
r2 = r2_score(y_actual, y_predicted)

print("R2:", r2)
~~~

**Expected output**
A numeric R² value.

**Explanation**
The metric compares model predictions with the mean-target baseline. Interpret it together with MAE and RMSE.

### 8.3 Understand a negative R² example

~~~python
poor_predictions = np.array([400, 50, 500, 20, 600, 10])

poor_r2 = r2_score(y_actual, poor_predictions)

print("Poor model R2:", poor_r2)
~~~

**Expected output**
A negative R² value for this deliberately poor prediction set.

**Explanation**
A negative score means this model performs worse than predicting the average actual target for every observation in this evaluation sample.

## 9. Compare regression metrics together

### 9.1 Build a metric summary

~~~python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

metric_summary = {
    "MAE": mean_absolute_error(y_actual, y_predicted),
    "MSE": mean_squared_error(y_actual, y_predicted),
    "RMSE": np.sqrt(mean_squared_error(y_actual, y_predicted)),
    "R2": r2_score(y_actual, y_predicted)
}

print(metric_summary)
~~~

**Expected output**
A dictionary containing MAE, MSE, RMSE and R².

**Explanation**
Each metric answers a different question. Do not select a model from one number without considering the business target.

### 9.2 Compare two models

~~~python
model_a_predictions = np.array([105, 145, 195, 255, 295, 355])
model_b_predictions = np.array([90, 160, 180, 280, 260, 370])

models = {
    "Model A": model_a_predictions,
    "Model B": model_b_predictions
}

comparison_rows = []

for model_name, predictions in models.items():
    comparison_rows.append({
        "model": model_name,
        "MAE": mean_absolute_error(y_actual, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_actual, predictions)),
        "R2": r2_score(y_actual, predictions)
    })

model_comparison = pd.DataFrame(comparison_rows)
print(model_comparison)
~~~

**Expected output**
A two-row comparison table containing regression metrics for both candidate models.

**Explanation**
Both models must be evaluated on the same observations. Compare error size, large-error sensitivity and baseline-relative performance.

## 10. Visualize actual versus predicted values

### 10.1 Plot actual and predicted values

~~~python
plt.plot(y_actual, marker="o", label="Actual")
plt.plot(y_predicted, marker="x", label="Predicted")
plt.xlabel("Observation")
plt.ylabel("Target value")
plt.title("Actual versus Predicted")
plt.legend()
plt.show()
~~~

**Expected output**
A line chart comparing actual and predicted values for each observation.

**Explanation**
The chart can reveal whether predictions follow the general movement of the actual target.

### 10.2 Plot residuals

~~~python
residuals = y_actual - y_predicted

plt.scatter(y_predicted, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted value")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()
~~~

**Expected output**
A scatter plot of residuals with a horizontal zero reference line.

**Explanation**
Look for systematic patterns, changing spread or unusual observations. Random-looking residuals are generally more reassuring than a strong visible pattern, but interpretation depends on the task.

### 10.3 Plot residual distribution

~~~python
plt.hist(residuals, bins=6)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()
~~~

**Expected output**
A histogram showing how residuals are distributed around zero.

**Explanation**
A strongly shifted distribution may indicate systematic overprediction or underprediction.

## 11. How to select a regression metric

| Situation | Useful metric | Why |
|---|---|---|
| Easy business explanation of typical error | MAE | Same unit as target and less sensitive than squared error |
| Large errors are especially costly | RMSE | Large errors have more influence |
| Comparing against mean-target baseline | R² | Shows relative performance against a simple baseline |
| Target has extreme outliers | MAE alongside robust residual analysis | RMSE may be dominated by extreme errors |
| Forecasting revenue or demand | MAE/RMSE plus segment and time analysis | Business impact depends on scale and direction |

Always compare with a simple baseline. A complex model that does not improve on the baseline may not justify its operational cost.

## 12. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Treat R² as accuracy | Explain it as performance relative to a mean baseline |
| Compare MSE values with different target units | Compare models on the same target scale or use an appropriate normalized metric |
| Ignore outliers | Inspect residuals and source records |
| Report only RMSE | Include MAE and business interpretation |
| Assume negative R² is impossible | Explain that test performance can be worse than the mean baseline |
| Forget whether residual means actual minus predicted | State the convention clearly |
| Evaluate training predictions only | Use held-out evaluation data |
| Compare models on different test rows | Use the same evaluation observations |

## 13. Interview-ready explanations

**What is MAE?**
- Average absolute prediction error.
- Easy to explain in the target's original unit.
- Less sensitive to large errors than squared-error metrics.

**What is MSE?**
- Average squared prediction error.
- Penalizes large errors more strongly.
- Expressed in squared target units.

**What is RMSE?**
- Square root of MSE.
- Expressed in the target's original unit.
- Gives larger errors more influence.

**What is R²?**
- Compares model predictions with a mean-target baseline.
- Can be negative on unseen data.
- Does not directly describe the size of an error in business units.

**Why use MAE and RMSE together?**
- MAE describes typical absolute error.
- RMSE highlights the influence of larger errors.
- A large gap between them can indicate some unusually large errors.

## 14. Scenario-based interview practice

### Scenario 1: RMSE is much higher than MAE
**Expected response:**
- A small number of large errors may be influencing RMSE.
- Inspect residuals and the records with largest absolute errors.
- Check data quality, target distribution and important subgroups.
- Do not remove outliers without a valid business reason.

### Scenario 2: R² is negative on test data
**Expected response:**
- The model performs worse than the mean-target baseline on this test set.
- Check train-test distribution differences, preprocessing, target alignment and overfitting.
- Compare with a simple baseline and review residuals.

### Scenario 3: Model A has lower MAE but Model B has lower RMSE
**Expected response:**
- Model A may have smaller typical absolute errors.
- Model B may avoid some large errors or distribute errors differently.
- Inspect residuals and business cost of large mistakes.
- Choose the metric aligned with the use case and validate by segment.

### Scenario 4: A model has high R² but poor business usefulness
**Expected response:**
- Check MAE and RMSE in business units.
- Inspect errors for important high-value or low-value segments.
- Check systematic overprediction or underprediction.
- Confirm that the evaluation target and time period represent the real decision.

## 15. Lesson completion checklist

- [ ] I can explain actual values, predictions and residuals.
- [ ] I can calculate MAE, MSE, RMSE and R².
- [ ] I can explain the business meaning of each metric.
- [ ] I understand why RMSE is sensitive to large errors.
- [ ] I understand why R² can be negative.
- [ ] I can compare two regression models fairly.
- [ ] I can interpret residual plots.
- [ ] I can explain regression metrics in an interview.

**Next lesson:** Day 48 — Cross-Validation and Hyperparameter Tuning.
