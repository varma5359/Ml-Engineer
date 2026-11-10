# Day 19: Linear Regression, Assumptions and Equation

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain regression and linear regression in simple language.
- Identify the numerical target in a regression problem.
- Understand features, coefficients and intercept without memorizing equations.
- Train a simple linear regression model using scikit-learn.
- Generate predictions for new observations.
- Interpret coefficient direction and units carefully.
- Visualize a fitted relationship.
- Inspect residuals and common assumptions.
- Recognize when a linear model may be unsuitable.

## 2. What is regression?

**Regression** is a supervised machine-learning task used to predict a numerical value.

Examples:
- Predict house price.
- Estimate delivery time.
- Forecast monthly sales.
- Predict electricity consumption.
- Estimate insurance claim amount.

The target is a measurable quantity rather than a category label.

## 3. What is linear regression?

**Linear regression** is a model that estimates a numerical target by combining input features with learned weights and an intercept.

In simple terms:
- The model receives feature values.
- It learns how each feature is associated with the target.
- It combines the feature contributions.
- It produces a numerical prediction.

For a one-feature example, the model learns a straight-line relationship between the input and target. With multiple features, it combines the contributions from several inputs.

**Important:** A coefficient describes an association within the fitted model. It does not prove that changing the feature causes the target to change.

## 4. Understand the important terms

| Term | Beginner-friendly meaning | Example |
|---|---|---|
| Feature | Input information used by the model | House area |
| Target | Numerical value to predict | House price |
| Coefficient | Learned weight for a feature | How the prediction changes with area in the fitted model |
| Intercept | Model's baseline prediction when feature inputs are zero | Starting value in the fitted model |
| Prediction | Model's estimated output | Estimated house price |
| Residual | Difference between actual and predicted target | Actual price compared with predicted price |
| Training | Learning model parameters from data | Fitting LinearRegression |
| Inference | Using the fitted model to predict new observations | Predict price for a new house |

## 5. Create a simple house-price dataset

### 5.1 Import libraries

### 5.1. Import Pandas, NumPy and plotting libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries ready")
```

**Expected output**

```text
Libraries ready
```

**Explanation**

Pandas manages the table, NumPy supports numerical work, and Matplotlib/Seaborn create charts.



### 5.2 Create a small house dataset

### 5.2. Create area and price observations

```python
df = pd.DataFrame({
    "area_sqft": [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
                  1600, 1700, 1800, 1900, 2000],
    "price_lakhs": [20, 23, 25, 29, 31, 34, 37, 40, 43, 46,
                    49, 52, 56, 59, 62]
})

print(df.head())
```

**Expected output**

```text
The first five house records
```

**Explanation**

Each row is one house. Area is the feature and price in lakhs is the numerical target. The values are illustrative.



### 5.3 Inspect the dataset

### 5.3. Check shape and types

```python
print(df.shape)
print(df.dtypes)
```

**Expected output**

```text
(15, 2), followed by numeric dtypes
```

**Explanation**

There are 15 observations and two columns. Both columns are numerical.



### 5.4 Inspect summary statistics

### 5.4. Use describe()

```python
print(df.describe())
```

**Expected output**

```text
Count, mean, standard deviation, minimum, quartiles and maximum
```

**Explanation**

This provides an initial view of the feature and target ranges.



### 5.5 Visualize the relationship

### 5.5. Create a scatter plot

```python
sns.scatterplot(data=df, x="area_sqft", y="price_lakhs")
plt.xlabel("Area (square feet)")
plt.ylabel("Price (lakhs)")
plt.title("House area versus price")
plt.show()
```

**Expected output**

```text
A scatter plot with an upward pattern
```

**Explanation**

The plot suggests a positive relationship in this constructed sample. A visual pattern alone does not prove a causal relationship.



## 6. Separate features and target

### 6.1 Select the feature table

### 6.1. Create X

```python
X = df[["area_sqft"]]

print(X.head())
```

**Expected output**

```text
A one-column DataFrame containing area_sqft
```

**Explanation**

Scikit-learn expects the feature input as a two-dimensional structure for ordinary tabular estimators. Double brackets select a DataFrame.



### 6.2 Select the target

### 6.2. Create y

```python
y = df["price_lakhs"]

print(y.head())
```

**Expected output**

```text
A Series containing house prices
```

**Explanation**

The target is one-dimensional. It contains the actual numerical values that the model will learn to predict.



### 6.3 Check the shapes

### 6.3. Verify feature and target dimensions

```python
print(X.shape)
print(y.shape)
print(X.shape[0] == y.shape[0])
```

**Expected output**

```text
(15, 1), (15,), True
```

**Explanation**

There is one target value for each feature row. Also preserve the row alignment, not only the row count.



## 7. Split training and test data

A model should be evaluated on observations it did not use to learn its parameters.

### 7.1 Import train_test_split

### 7.1. Import the split function

```python
from sklearn.model_selection import train_test_split

print("Split function ready")
```

**Expected output**

```text
Split function ready
```

**Explanation**

The function creates training and test partitions.



### 7.2 Split X and y together

### 7.2. Create a train-test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

**Expected output**

```text
(12, 1), (3, 1), (12,), (3,)
```

**Explanation**

The model will learn from 12 houses and be evaluated on three held-out houses. The split keeps each feature row paired with its target.



### 7.3 Inspect training observations

### 7.3. Display training rows

```python
print(X_train.join(y_train))
```

**Expected output**

```text
Training area and price pairs
```

**Explanation**

This lets you inspect which observations are used to fit the model.



### 7.4 Inspect test observations

### 7.4. Display test rows

```python
print(X_test.join(y_test))
```

**Expected output**

```text
Held-out area and price pairs
```

**Explanation**

Do not use test values to fit the model. They are reserved for evaluation.



## 8. Train a Linear Regression model

### 8.1 Import LinearRegression

### 8.1. Import the estimator

```python
from sklearn.linear_model import LinearRegression

print("LinearRegression imported")
```

**Expected output**

```text
LinearRegression imported
```

**Explanation**

This estimator fits a linear relationship between the input features and a numerical target.



### 8.2 Create the model object

### 8.2. Initialize LinearRegression

```python
model = LinearRegression()

print(model)
```

**Expected output**

```text
LinearRegression()
```

**Explanation**

The object is initialized but has not learned any coefficients yet.



### 8.3 Fit the model

### 8.3. Train the model

```python
model.fit(X_train, y_train)

print("Training completed")
```

**Expected output**

```text
Training completed
```

**Explanation**

The fit method learns model parameters using the training feature values and target values.



### 8.4 Inspect the learned coefficient

### 8.4. Read the coefficient

```python
print(model.coef_)
```

**Expected output**

```text
A one-element numerical array
```

**Explanation**

The coefficient describes the fitted association between area and predicted price. Its unit is price-lakhs per square foot in this example. Interpret it only within the model and data context.



### 8.5 Inspect the intercept

### 8.5. Read the intercept

```python
print(model.intercept_)
```

**Expected output**

```text
A numerical intercept value
```

**Explanation**

The intercept is the model's predicted starting value when the input feature is zero. Zero square feet is outside the meaningful range of this house dataset, so the intercept may not have a practical business interpretation.



### 8.6 Inspect model parameters together

### 8.6. Display coefficient and intercept

```python
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
```

**Expected output**

```text
The learned coefficient and intercept
```

**Explanation**

These values define the fitted straight-line relationship. Do not interpret the coefficient as a causal effect.



## 9. Generate predictions

### 9.1 Predict on training data

### 9.1. Generate training predictions

```python
train_predictions = model.predict(X_train)

print(train_predictions[:5])
```

**Expected output**

```text
The first five predicted prices for training observations
```

**Explanation**

These predictions help inspect how the fitted model represents the data it learned from.



### 9.2 Predict on test data

### 9.2. Generate test predictions

```python
test_predictions = model.predict(X_test)

print(test_predictions)
```

**Expected output**

```text
Predicted prices for the held-out houses
```

**Explanation**

The model uses the learned relationship to estimate prices for unseen feature values.



### 9.3 Compare actual and predicted values

### 9.3. Create an actual-versus-predicted table

```python
prediction_table = pd.DataFrame({
    "area_sqft": X_test["area_sqft"].values,
    "actual_price": y_test.values,
    "predicted_price": test_predictions
})

print(prediction_table)
```

**Expected output**

```text
A table containing area, actual price and predicted price
```

**Explanation**

The table helps inspect individual prediction errors. A later lesson covers detailed error metrics.



### 9.4 Predict one new house

### 9.4. Predict price for a new area

```python
new_house = pd.DataFrame({"area_sqft": [1250]})

new_prediction = model.predict(new_house)

print(new_prediction[0])
```

**Expected output**

```text
A numerical price prediction
```

**Explanation**

The new input is a DataFrame with the same feature name and structure used during training.



### 9.5 Predict several new houses

### 9.5. Predict multiple new observations

```python
new_houses = pd.DataFrame({
    "area_sqft": [750, 1250, 1750]
})

new_predictions = model.predict(new_houses)

print(new_predictions)
```

**Expected output**

```text
Three predicted house prices
```

**Explanation**

The model can process multiple rows in one call. Feature names and order should match the training schema.



## 10. Visualize the fitted regression line

### 10.1 Plot training observations

### 10.1. Plot training data

```python
plt.scatter(X_train["area_sqft"], y_train)
plt.xlabel("Area (square feet)")
plt.ylabel("Price (lakhs)")
plt.title("Training observations")
plt.show()
```

**Expected output**

```text
A scatter plot of training observations
```

**Explanation**

Only training observations are shown. The test set remains held out.



### 10.2 Sort feature values for a smooth line

### 10.2. Sort training rows for plotting

```python
plot_data = X_train.copy()
plot_data["actual_price"] = y_train
plot_data = plot_data.sort_values("area_sqft")

print(plot_data.head())
```

**Expected output**

```text
Training observations ordered by area
```

**Explanation**

Sorting makes the predicted line display from smaller to larger area rather than jumping between unsorted observations.



### 10.3 Generate line predictions

### 10.3. Predict along the sorted feature values

```python
line_predictions = model.predict(plot_data[["area_sqft"]])

print(line_predictions[:5])
```

**Expected output**

```text
The first five fitted values in ascending area order
```

**Explanation**

These predictions form the fitted straight-line pattern.



### 10.4 Draw observations and fitted line

### 10.4. Plot the fitted relationship

```python
plt.scatter(X_train["area_sqft"], y_train, label="Training observations")
plt.plot(plot_data["area_sqft"], line_predictions, label="Fitted line")
plt.xlabel("Area (square feet)")
plt.ylabel("Price (lakhs)")
plt.title("Linear regression fit")
plt.legend()
plt.show()
```

**Expected output**

```text
A scatter plot with an upward fitted line
```

**Explanation**

The line summarizes the relationship learned from training data. It does not need to pass through every point.



## 11. Understand coefficients carefully

A coefficient describes how the model's prediction changes when a feature changes while the other included features are held constant.

For a simple one-feature model:
- A positive coefficient means larger feature values are associated with higher predictions.
- A negative coefficient means larger feature values are associated with lower predictions.
- A coefficient near zero means the fitted model assigns little linear change to that feature.

Interpretation depends on:
- Feature units.
- Feature scaling.
- Correlation with other features.
- Data range.
- Model specification.
- Whether the relationship is stable.

### 11.1 Change feature units

If area is converted from square feet to square metres, the coefficient's numerical value changes because the unit changes. The predicted prices should remain consistent when inputs are converted correctly.

### 11.2 Do not compare coefficients blindly

A coefficient of 2 for age and 0.5 for income does not automatically mean age is more important. The units and ranges may be completely different. Feature scaling and domain interpretation matter.

## 12. Multiple linear regression preview

Multiple linear regression uses more than one input feature. The model combines learned contributions from all included features.

### 12.1 Create a multi-feature dataset

### 12.1. Add bedroom count

```python
multi_df = pd.DataFrame({
    "area_sqft": [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
                  1600, 1700, 1800, 1900, 2000],
    "bedrooms": [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    "price_lakhs": [20, 22, 25, 27, 30, 33, 36, 39, 42, 45, 48, 52, 55, 58, 62]
})

print(multi_df.head())
```

**Expected output**

```text
A table containing area, bedrooms and price
```

**Explanation**

The additional feature may explain information beyond area. Its usefulness must be validated.



### 12.2 Select multiple features

### 12.2. Create a multi-feature X

```python
X_multi = multi_df[["area_sqft", "bedrooms"]]
y_multi = multi_df["price_lakhs"]

print(X_multi.shape)
```

**Expected output**

```text
(15, 2)
```

**Explanation**

Each row has two input features.



### 12.3 Split multi-feature data

### 12.3. Create training and test sets

```python
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi,
    y_multi,
    test_size=0.2,
    random_state=42
)

print(X_train_m.shape, X_test_m.shape)
```

**Expected output**

```text
(12, 2) and (3, 2)
```

**Explanation**

The model will learn using both features together.



### 12.4 Fit multiple linear regression

### 12.4. Train with two features

```python
multi_model = LinearRegression()

multi_model.fit(X_train_m, y_train_m)

print("Multiple-feature model fitted")
```

**Expected output**

```text
Multiple-feature model fitted
```

**Explanation**

The estimator learns one coefficient per input feature plus an intercept.



### 12.5 Inspect multiple coefficients

### 12.5. Read the coefficients

```python
coefficient_table = pd.DataFrame({
    "feature": X_multi.columns,
    "coefficient": multi_model.coef_
})

print(coefficient_table)
```

**Expected output**

```text
A two-row table with one coefficient per feature
```

**Explanation**

Each coefficient describes the model's fitted association while holding the other included feature constant. Correlated features can make individual coefficients unstable.



### 12.6 Inspect feature names and coefficients together

### 12.6. Print model feature names

```python
print(multi_model.feature_names_in_)
print(multi_model.coef_)
```

**Expected output**

```text
The feature names followed by their coefficients
```

**Explanation**

Keep feature names aligned with coefficient order. Do not reorder one without the other.



## 13. Assumptions and diagnostics

Linear regression is a useful starting model, but the suitability of a linear model depends on the data and objective.

Common considerations:
- The relationship between inputs and target should be reasonably represented by the chosen linear form.
- Observations and errors should be independent for the relevant interpretation.
- Residual spread should be reasonably stable for classical inference.
- Strongly influential observations should be investigated.
- Multicollinearity can make individual coefficients unstable.
- For some statistical inference tasks, residual distribution assumptions matter. They are not a universal requirement for predictive use.

### 13.1 Calculate training residuals

### 13.1. Create residual values

```python
train_residuals = y_train - train_predictions

print(train_residuals.head())
```

**Expected output**

```text
Residual values for the first training observations
```

**Explanation**

A residual is actual target minus predicted target. Positive residuals mean the model predicted too low; negative residuals mean it predicted too high.



### 13.2 Plot residuals against predictions

### 13.2. Create a residual plot

```python
plt.scatter(train_predictions, train_residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted price")
plt.ylabel("Residual")
plt.title("Residuals versus predicted price")
plt.show()
```

**Expected output**

```text
A scatter plot of residuals with a horizontal zero reference line
```

**Explanation**

A random-looking cloud around zero is often a useful sign. Curves, funnels or clusters may indicate patterns the model has not captured.



### 13.3 Plot residual distribution

### 13.3. Inspect residual histogram

```python
sns.histplot(train_residuals, bins=8, kde=True)
plt.xlabel("Residual")
plt.title("Training residual distribution")
plt.show()
```

**Expected output**

```text
A histogram of training residuals
```

**Explanation**

This helps inspect residual shape. Do not treat a bell-shaped histogram alone as proof that all regression assumptions are satisfied.



### 13.4 Compare residuals by feature range

### 13.4. Add prediction bands for inspection

```python
residual_view = pd.DataFrame({
    "area_sqft": X_train["area_sqft"].values,
    "actual": y_train.values,
    "predicted": train_predictions,
    "residual": train_residuals.values
})

print(residual_view.sort_values("area_sqft"))
```

**Expected output**

```text
A table of area, actual price, predicted price and residual
```

**Explanation**

Sorting helps inspect whether error direction changes across the feature range.



## 14. When linear regression may be unsuitable

Investigate alternatives when:
- The relationship is strongly curved.
- Important features are missing.
- Errors vary systematically across groups.
- A few influential observations dominate the fit.
- The target is categorical rather than numerical.
- Predictions outside the training range are unreliable.
- The data has time or group structure that the split/model ignores.

A more complex algorithm is not automatically better. Start with a simple model, inspect errors and compare alternatives using fair validation.

## 15. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Use a category as a numeric measurement without encoding | The model interprets arbitrary numeric order | Encode categories appropriately |
| Train and evaluate on the same rows | Performance is optimistic | Use held-out evaluation |
| Interpret a coefficient as causation | Association is not causal proof | State the model context and limitations |
| Ignore feature units | Coefficients become misleading | Document units and scaling |
| Ignore residual patterns | Systematic errors remain hidden | Inspect residuals against predictions and features |
| Extrapolate far beyond training range | Relationship may not continue | Check whether prediction is within supported range |
| Include post-outcome features | Target leakage | Verify prediction-time availability |
| Assume all residuals must be normal for prediction | Overstates a classical inference assumption | Match assumptions to the modelling objective |

## 16. Interview-ready explanations

**What is linear regression?**  
It is a supervised regression model that predicts a numerical target by combining input features with learned weights and an intercept.

**What is a coefficient?**  
It is a learned weight that describes how the model's prediction changes with a feature, holding other included features constant. Interpretation depends on units, scaling and other features.

**What is an intercept?**  
It is the model's baseline prediction when input features are zero. It may not have a meaningful real-world interpretation if zero is outside the observed feature range.

**Why inspect residuals?**  
Residuals show where predictions are too high or too low and can reveal systematic patterns, changing error spread or influential observations.

**Does a positive coefficient prove a feature causes the target to increase?**  
No. It describes a fitted association in the model, not a causal effect.

## 17. Scenario questions

1. A house-price model predicts premium properties too low. What do you inspect?  
   **Answer:** Residuals by price range, missing premium-property features, sample representation and whether a straight-line relationship is sufficient.

2. A coefficient is positive. Does it prove the feature causes price to rise?  
   **Answer:** No. It is an association within the fitted model.

3. Training predictions are accurate but test predictions are poor. What might be happening?  
   **Answer:** Overfitting, distribution mismatch, leakage or an inappropriate split.

4. A coefficient changes significantly when a correlated feature is added. What might explain it?  
   **Answer:** Multicollinearity or overlapping information between predictors.

5. The model predicts for a house area far beyond all training examples. What is the concern?  
   **Answer:** Extrapolation. The learned relationship may not remain valid outside the observed range.

## 18. Lesson completion checklist

- [ ] I can explain regression and linear regression.
- [ ] I can identify features and numerical target.
- [ ] I understand coefficient and intercept in plain language.
- [ ] I can split data and fit LinearRegression.
- [ ] I can generate predictions for new rows.
- [ ] I can visualize the fitted relationship.
- [ ] I can calculate and interpret residuals.
- [ ] I can inspect residual plots.
- [ ] I can explain limitations and common assumptions.
- [ ] I avoid causal claims from coefficients.

**Next lesson:** Day 20 — Cost Functions and Residual Analysis.
