# Day 20: Cost Functions and Residual Analysis

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain prediction error, residual and loss in simple language.
- Distinguish an individual residual from an overall metric.
- Understand why a training algorithm needs an objective or loss.
- Calculate and interpret MAE, MSE, RMSE and R-squared using scikit-learn.
- Compare metrics in the target's business units.
- Inspect residual patterns and large errors.
- Identify underprediction, overprediction and changing error spread.
- Choose evaluation metrics based on business impact.
- Avoid relying on one metric alone.

## 2. What is prediction error?

A prediction is an estimate produced by a model. The actual target is the observed value.

The **prediction error** describes the difference between the actual value and the prediction. For regression, the **residual** is commonly defined as actual target minus predicted target.

Interpretation:
- Positive residual: the model predicted too low.
- Negative residual: the model predicted too high.
- Zero residual: the prediction matches the actual value.

Example:

| Actual delivery time | Predicted delivery time | Residual |
|---:|---:|---:|
| 30 min | 25 min | +5 min |
| 40 min | 45 min | -5 min |
| 50 min | 50 min | 0 min |

## 3. What is a loss function or cost function?

A **loss function** assigns a numerical value to how wrong a model's predictions are. Training algorithms use an objective to adjust model parameters and reduce the selected loss.

In everyday terms:
- The model makes predictions.
- The loss measures prediction mistakes.
- The training algorithm adjusts model parameters.
- The process seeks a model with lower training loss.

Different losses penalize errors differently. A squared-error objective gives large errors more influence than an absolute-error objective.

The best loss depends on the problem, the algorithm and the business cost of different mistakes.

## 4. Create an example prediction dataset

### 4.1 Import libraries

### 4.1. Import NumPy, Pandas and metrics

```python
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Metric tools ready")
```

**Expected output**

```text
Metric tools ready
```

**Explanation**

The metrics are provided by scikit-learn. NumPy is used to calculate the square root of MSE.



### 4.2 Create actual and predicted values

### 4.2. Prepare a small delivery-time example

```python
actual = np.array([30, 40, 50, 60, 70])
predicted = np.array([25, 45, 50, 55, 80])

print(actual)
print(predicted)
```

**Expected output**

```text
Actual values [30 40 50 60 70] and predicted values [25 45 50 55 80]
```

**Explanation**

Each position represents the same delivery observation in both arrays. Alignment is essential.



### 4.3 Build an inspection table

### 4.3. Create actual and predicted columns

```python
results = pd.DataFrame({
    "actual": actual,
    "predicted": predicted
})

print(results)
```

**Expected output**

```text
A five-row table containing actual and predicted values
```

**Explanation**

The table makes individual predictions easier to inspect.



## 5. Calculate residuals

### 5.1 Calculate residual values

### 5.1. Subtract prediction from actual

```python
residuals = actual - predicted

print(residuals)
```

**Expected output**

```text
[ 5 -5  0  5 -10]
```

**Explanation**

The residual is actual minus predicted. Positive values mean underprediction and negative values mean overprediction.



### 5.2 Add residuals to the table

### 5.2. Create a residual column

```python
results["residual"] = residuals

print(results)
```

**Expected output**

```text
A table containing actual, predicted and residual columns
```

**Explanation**

This makes the direction and size of each prediction error visible.



### 5.3 Calculate absolute residuals

### 5.3. Inspect error magnitude

```python
results["absolute_error"] = np.abs(results["residual"])

print(results)
```

**Expected output**

```text
A table with non-negative absolute errors
```

**Explanation**

Absolute error ignores direction and shows the magnitude of each mistake.



### 5.4 Calculate squared residuals

### 5.4. Inspect squared errors

```python
results["squared_error"] = results["residual"] ** 2

print(results)
```

**Expected output**

```text
A table containing squared residuals
```

**Explanation**

Squaring makes all errors non-negative and gives larger errors much greater influence.



## 6. Mean Absolute Error (MAE)

**Mean Absolute Error (MAE)** reports the average absolute difference between actual and predicted values.

Plain-language interpretation:
- It is measured in the same units as the target.
- It is easy to communicate to stakeholders.
- Each absolute error contributes linearly to the average.
- Large errors matter, but they do not receive the same extra squared penalty as MSE.

### 6.1 Calculate MAE

### 6.1. Use mean_absolute_error()

```python
mae = mean_absolute_error(actual, predicted)

print(mae)
```

**Expected output**

```text
5.0
```

**Explanation**

The average absolute prediction error is five minutes for this example.



### 6.2 Confirm MAE using the error column

### 6.2. Calculate mean absolute residual

```python
manual_mae = results["absolute_error"].mean()

print(manual_mae)
```

**Expected output**

```text
5.0
```

**Explanation**

The mean of the absolute residuals matches the scikit-learn MAE result.



### 6.3 Explain MAE to a business stakeholder

If a delivery-time model has MAE of five minutes, the average absolute difference between its estimates and actual delivery times is five minutes on the evaluated observations.

It does not mean every prediction is wrong by exactly five minutes. Some errors may be smaller and others larger.

## 7. Mean Squared Error (MSE)

**Mean Squared Error (MSE)** averages squared prediction errors.

Key points:
- It is measured in squared target units.
- Large errors receive a stronger penalty.
- It is useful for optimization and comparing models under the same evaluation setup.
- Its numerical value is less intuitive for non-technical stakeholders because the units are squared.

### 7.1 Calculate MSE

### 7.1. Use mean_squared_error()

```python
mse = mean_squared_error(actual, predicted)

print(mse)
```

**Expected output**

```text
35.0
```

**Explanation**

The metric averages the squared residuals. The result is in squared minutes for this delivery example.



### 7.2 Confirm MSE from the squared-error column

### 7.2. Calculate the mean squared residual

```python
manual_mse = results["squared_error"].mean()

print(manual_mse)
```

**Expected output**

```text
35.0
```

**Explanation**

This matches scikit-learn's result for the same aligned observations.



### 7.3 Why does MSE emphasize large errors?

Compare two possible mistakes:
- Error of 2 units.
- Error of 10 units.

The larger error contributes much more after squaring. This makes MSE sensitive to large prediction mistakes.

## 8. Root Mean Squared Error (RMSE)

**Root Mean Squared Error (RMSE)** is the square root of MSE. It returns the metric to the same unit as the target.

### 8.1 Calculate RMSE

### 8.1. Calculate the square root of MSE

```python
rmse = np.sqrt(mse)

print(rmse)
```

**Expected output**

```text
Approximately 5.916
```

**Explanation**

The result is expressed in minutes, making it easier to compare with the target's scale.



### 8.2 Use scikit-learn's root_mean_squared_error when available

### 8.2. Use the direct RMSE metric when supported

```python
try:
    from sklearn.metrics import root_mean_squared_error
    rmse_direct = root_mean_squared_error(actual, predicted)
except ImportError:
    rmse_direct = np.sqrt(mean_squared_error(actual, predicted))

print(rmse_direct)
```

**Expected output**

```text
Approximately 5.916
```

**Explanation**

Recent scikit-learn versions provide a direct function. If this import is unavailable in an older environment, calculate the square root of mean_squared_error instead.



### 8.3 Compare MAE and RMSE

### 8.3. Display MAE and RMSE together

```python
print("MAE:", mae)
print("RMSE:", rmse)
```

**Expected output**

```text
MAE: 5.0 and RMSE: approximately 5.916
```

**Explanation**

RMSE is larger than MAE in this example because squaring gives larger errors more influence.



## 9. R-squared

**R-squared** compares a regression model's prediction performance with a simple reference based on the target's mean.

Plain-language interpretation:
- A value closer to one indicates the model explains more variation relative to that reference on the evaluated data.
- A value around zero indicates performance similar to the mean-reference comparison.
- A negative value means the model is worse than that reference on the evaluated data.

R-squared does not tell you whether predictions are unbiased, whether errors are acceptable in business units or whether the model generalizes to future data.

### 9.1 Calculate R-squared

### 9.1. Use r2_score()

```python
r2 = r2_score(actual, predicted)

print(r2)
```

**Expected output**

```text
An R-squared value for the example predictions
```

**Explanation**

The exact value is determined by the actual and predicted arrays. Interpret it alongside error metrics and residual plots.



### 9.2 Compare actual and predicted values

### 9.2. Display the prediction table

```python
print(results[["actual", "predicted", "residual"]])
```

**Expected output**

```text
A table of actual, predicted and residual values
```

**Explanation**

A single R-squared value cannot show which individual observations were over- or underpredicted.



## 10. Compare all common regression metrics

### 10.1 Create a metric summary

### 10.1. Build a regression metric table

```python
metric_summary = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2]
})

print(metric_summary)
```

**Expected output**

```text
A four-row table of regression metrics
```

**Explanation**

The table is a compact summary. Explain the unit and interpretation of each metric when reporting it.



### 10.2 Understand the metrics

| Metric | What it describes | Unit | Sensitivity to large errors |
|---|---|---|---|
| MAE | Average absolute error | Same as target | Linear |
| MSE | Average squared error | Squared target unit | Strong |
| RMSE | Square root of average squared error | Same as target | Strong |
| R-squared | Performance relative to mean-target reference | No target unit | Depends on residual behaviour |

## 11. Residual analysis

A residual plot helps detect patterns that an overall metric can hide.

Look for:
- Residuals mostly above zero: possible underprediction.
- Residuals mostly below zero: possible overprediction.
- Curved pattern: the relationship may not be captured by a straight-line model.
- Funnel-shaped spread: error variation may change across prediction range.
- Clusters: segments may behave differently.
- Isolated large residuals: unusual observations or difficult cases.

### 11.1 Plot residuals against predictions

### 11.1. Create a residual scatter plot

```python
import matplotlib.pyplot as plt

plt.scatter(predicted, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted delivery time (minutes)")
plt.ylabel("Residual (minutes)")
plt.title("Residuals versus predictions")
plt.show()
```

**Expected output**

```text
A residual scatter plot with a zero reference line
```

**Explanation**

The horizontal reference helps identify overprediction and underprediction. Patterns suggest that the model may be missing structure.



### 11.2 Plot residual distribution

### 11.2. Create a residual histogram

```python
import seaborn as sns

sns.histplot(residuals, bins=6, kde=True)
plt.xlabel("Residual (minutes)")
plt.title("Residual distribution")
plt.show()
```

**Expected output**

```text
A histogram of residuals
```

**Explanation**

The histogram shows the distribution of prediction errors. It should be interpreted with the residual plot and business context.



### 11.3 Compare residuals against a feature

### 11.3. Plot residuals by actual delivery time

```python
plt.scatter(actual, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Actual delivery time (minutes)")
plt.ylabel("Residual (minutes)")
plt.title("Residuals versus actual values")
plt.show()
```

**Expected output**

```text
A scatter plot of residuals against actual values
```

**Explanation**

This can reveal whether the model behaves differently for short and long deliveries.



### 11.4 Inspect the largest absolute errors

### 11.4. Sort observations by error magnitude

```python
results_sorted = results.sort_values(
    "absolute_error",
    ascending=False
)

print(results_sorted)
```

**Expected output**

```text
Rows ordered from largest to smallest absolute error
```

**Explanation**

Review the largest errors first. They may be valid unusual cases, data issues or evidence of missing predictors.



### 11.5 Calculate mean residual

### 11.5. Inspect average residual direction

```python
print(results["residual"].mean())
```

**Expected output**

```text
A signed average residual
```

**Explanation**

A positive mean indicates average underprediction in this sample; a negative mean indicates average overprediction. Positive and negative errors can cancel, so inspect the full distribution too.



## 12. Compare models with different error patterns

Two models may have similar MAE but different RMSE because one model has a few very large errors.

### 12.1 Create two prediction sets

### 12.1. Prepare candidate model predictions

```python
actual_compare = np.array([20, 30, 40, 50, 60, 70])

pred_model_a = np.array([22, 28, 42, 48, 62, 68])
pred_model_b = np.array([20, 30, 40, 50, 60, 90])

print(actual_compare)
print(pred_model_a)
print(pred_model_b)
```

**Expected output**

```text
Three arrays showing actual values and two prediction sets
```

**Explanation**

Model A has several small errors. Model B is exact on most rows but has one larger error.



### 12.2 Calculate MAE for both models

### 12.2. Compare absolute errors

```python
mae_a = mean_absolute_error(actual_compare, pred_model_a)
mae_b = mean_absolute_error(actual_compare, pred_model_b)

print(mae_a, mae_b)
```

**Expected output**

```text
MAE values for model A and model B
```

**Explanation**

MAE summarizes average absolute error. It does not show the complete error distribution.



### 12.3 Calculate RMSE for both models

### 12.3. Compare squared-error sensitivity

```python
rmse_a = np.sqrt(mean_squared_error(actual_compare, pred_model_a))
rmse_b = np.sqrt(mean_squared_error(actual_compare, pred_model_b))

print(rmse_a, rmse_b)
```

**Expected output**

```text
RMSE values for model A and model B
```

**Explanation**

The large error in model B receives a stronger penalty, so its RMSE may be substantially higher.



### 12.4 Choose a metric based on business costs

If one very large prediction error can cause a major business loss, a metric that emphasizes large errors may be important. If the business wants a typical absolute error in understandable units, MAE may be easier to communicate.

Do not choose a metric only because it produces a favourable score. Agree on the metric based on the decision and error costs.

## 13. Segment-level residual analysis

An overall metric may hide systematic problems in a particular segment.

Examples:
- Luxury properties are underpredicted.
- Rural deliveries have larger errors.
- A certain product category has inconsistent forecasts.
- New customers have larger prediction errors than long-term customers.

### 13.1 Create a segment example

### 13.1. Create actual, predicted and region data

```python
segment_df = pd.DataFrame({
    "region": ["North", "North", "North", "South", "South", "South"],
    "actual": [30, 40, 50, 35, 45, 55],
    "predicted": [28, 38, 48, 25, 35, 45]
})

segment_df["residual"] = (
    segment_df["actual"] - segment_df["predicted"]
)

print(segment_df)
```

**Expected output**

```text
A table with region, actual, predicted and residual
```

**Explanation**

The residual column lets us compare prediction direction by region.



### 13.2 Calculate MAE by segment

### 13.2. Summarize segment errors

```python
segment_mae = {}

for region, group in segment_df.groupby("region"):
    segment_mae[region] = mean_absolute_error(
        group["actual"],
        group["predicted"]
    )

print(segment_mae)
```

**Expected output**

```text
A mean absolute error value for each region
```

**Explanation**

This uses a small teaching dataset. Depending on Pandas version, groupby.apply options may differ; an explicit loop is an alternative. Always report group counts alongside the metric.



### 13.3 Report segment counts and MAE

### 13.3. Create a segment report

```python
segment_rows = []

for region, group in segment_df.groupby("region"):
    segment_rows.append({
        "region": region,
        "count": len(group),
        "MAE": mean_absolute_error(group["actual"], group["predicted"])
    })

segment_report = pd.DataFrame(segment_rows)

print(segment_report)
```

**Expected output**

```text
A report containing region, sample count and MAE
```

**Explanation**

Counts provide context. A metric based on only a few records may be unstable.



## 14. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Report only one metric | Hides error patterns | Compare MAE, RMSE, R-squared and residuals |
| Confuse residual sign | Wrong interpretation of model bias | Define residual as actual minus predicted |
| Treat MSE as target units | MSE uses squared target units | Explain its unit and use RMSE for target-scale communication |
| Assume high R-squared means model is safe | It does not reveal all error patterns | Inspect residuals and business impact |
| Ignore segment errors | Aggregate scores can hide weak groups | Evaluate important segments |
| Compare metrics on different test sets | Comparison is unfair | Use the same evaluation observations |
| Tune repeatedly on test data | Test data influences development | Use validation or cross-validation |
| Ignore target scale | Error magnitude may be misunderstood | Report units and target context |

## 15. Interview-ready explanations

**What is a residual?**  
A residual is the actual target minus the model's prediction for one observation. A positive residual means underprediction; a negative residual means overprediction.

**What is MAE?**  
Mean Absolute Error is the average absolute prediction error. It is expressed in the same unit as the target and is relatively easy to explain.

**What is MSE?**  
Mean Squared Error averages squared prediction errors. Large errors receive more influence, and the result is in squared target units.

**What is RMSE?**  
Root Mean Squared Error is the square root of MSE. It is expressed in the target's units and gives large errors stronger influence than MAE.

**What does negative R-squared mean?**  
On evaluation data, the model performs worse than the mean-target reference used by the R-squared comparison.

**Why inspect residuals?**  
Residuals can reveal underprediction, overprediction, non-linear patterns, changing error spread and weak performance in particular segments.

## 16. Scenario questions

1. A delivery model has MAE of five minutes. How do you explain it?  
   **Answer:** Across evaluated deliveries, the average absolute difference between predictions and actual delivery times is five minutes.

2. RMSE is much larger than MAE. What might this indicate?  
   **Answer:** A few large prediction errors may be contributing strongly to RMSE.

3. A model has good overall MAE but consistently underpredicts luxury houses. What do you do?  
   **Answer:** Inspect residuals by price segment, check data representation and missing features, and validate any model change on suitable data.

4. Can R-squared be negative on test data?  
   **Answer:** Yes. It means the model performs worse than the mean-target reference under the R-squared comparison.

5. Which metric should a business use if very large errors are especially costly?  
   **Answer:** Consider RMSE or another loss that penalizes large errors, but select the metric based on actual business costs and inspect the error distribution.

## 17. Lesson completion checklist

- [ ] I can explain residual and loss in plain language.
- [ ] I can calculate residuals and absolute errors.
- [ ] I can calculate MAE, MSE and RMSE.
- [ ] I can interpret R-squared and negative values.
- [ ] I can compare model error patterns.
- [ ] I can create residual plots.
- [ ] I can inspect large errors and segment-level performance.
- [ ] I can choose metrics based on business costs.
- [ ] I avoid reporting only one metric.

**Next lesson:** Day 21 — Gradient Descent and Learning Rate.
