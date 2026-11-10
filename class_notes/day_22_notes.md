# Day 22: Multiple Linear Regression

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain multiple linear regression in simple language.
- Distinguish simple and multiple linear regression.
- Select several input features and one numerical target.
- Train a multiple linear regression model using scikit-learn.
- Interpret coefficients while considering units and other features.
- Generate predictions for new observations.
- Inspect feature correlation and coefficient stability.
- Evaluate residual patterns and segment-level errors.
- Recognize common issues such as multicollinearity, leakage and extrapolation.

## 2. What is multiple linear regression?

**Multiple linear regression** predicts a numerical target using more than one input feature.

Simple linear regression uses one feature, such as house area, to predict price.

Multiple linear regression may use:
- House area.
- Number of bedrooms.
- Property age.
- Distance from the city centre.
- Parking availability.

The model combines the contributions of the included features to produce one numerical prediction.

**Important:** A feature coefficient describes the model's fitted association while holding the other included features constant. It does not automatically describe a causal effect.

## 3. Business example — House-price prediction

A real-estate company wants to estimate house prices. Area alone may not explain every difference. Two houses with the same area can have different prices because of bedrooms, location, age or facilities.

The team can evaluate whether adding relevant features improves predictions compared with a simple area-only model.

## 4. Create a sample dataset

### 4.1 Import libraries

### 4.1. Import Pandas, NumPy and plotting libraries

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

Pandas and NumPy prepare data. Matplotlib and Seaborn help inspect feature relationships.



### 4.2 Create house records

### 4.2. Build a multi-feature dataset

```python
df = pd.DataFrame({
    "area_sqft": [650, 700, 800, 850, 900, 1000, 1100, 1150, 1200, 1300,
                  1350, 1450, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200],
    "bedrooms": [1, 1, 2, 2, 2, 2, 2, 3, 3, 3,
                 3, 3, 3, 3, 4, 4, 4, 4, 4, 5],
    "age_years": [15, 12, 10, 9, 8, 7, 6, 10, 5, 6,
                  4, 8, 3, 5, 6, 4, 3, 2, 3, 1],
    "distance_km": [12, 11, 10, 9, 8, 8, 7, 7, 6, 6,
                    5, 5, 4, 4, 3, 3, 2, 2, 1, 1],
    "price_lakhs": [22, 24, 27, 29, 31, 34, 37, 39, 42, 45,
                    47, 50, 53, 56, 60, 64, 68, 72, 77, 82]
})

print(df.head())
```

**Expected output**

```text
The first five house records
```

**Explanation**

The target is price_lakhs. The other columns are candidate features. This is an illustrative dataset, not a real market sample.



### 4.3 Inspect dataset shape

### 4.3. Check rows and columns

```python
print(df.shape)
```

**Expected output**

```text
(20, 5)
```

**Explanation**

There are 20 observations and five columns.



### 4.4 Inspect data types

### 4.4. Inspect dtypes

```python
print(df.dtypes)
```

**Expected output**

```text
Numeric types for all five columns
```

**Explanation**

All columns are numerical in this teaching example. Real datasets often contain categories that require preprocessing.



### 4.5 Inspect summary statistics

### 4.5. Use describe()

```python
print(df.describe().T)
```

**Expected output**

```text
A summary of all numeric columns
```

**Explanation**

The transpose displays each feature as a row. Inspect ranges and unusual values before modelling.



## 5. Understand the target and features

### 5.1 Select the target column

### 5.1. Create y from price_lakhs

```python
y = df["price_lakhs"]

print(y.head())
```

**Expected output**

```text
The first five house-price target values
```

**Explanation**

The target is the numerical quantity the model will learn to predict.



### 5.2 Select multiple features

### 5.2. Create X using several columns

```python
feature_columns = [
    "area_sqft",
    "bedrooms",
    "age_years",
    "distance_km"
]

X = df[feature_columns]

print(X.head())
```

**Expected output**

```text
A DataFrame containing four input features
```

**Explanation**

The feature list makes the input schema explicit. Do not include the target in X.



### 5.3 Inspect feature and target shapes

### 5.3. Check dimensions

```python
print(X.shape)
print(y.shape)
```

**Expected output**

```text
(20, 4) and (20,)
```

**Explanation**

There are four feature values per observation and one target value per row.



### 5.4 Check target and feature alignment

### 5.4. Compare observation counts

```python
print(X.shape[0] == y.shape[0])
```

**Expected output**

```text
True
```

**Explanation**

Matching row counts are necessary. The feature and target rows must also refer to the same houses in the same order.



## 6. Explore feature relationships

### 6.1 Inspect feature correlations

### 6.1. Calculate feature correlations

```python
print(X.corr().round(2))
```

**Expected output**

```text
A correlation matrix for the four features
```

**Explanation**

This can reveal relationships among predictors. High correlation may complicate coefficient interpretation in linear models.



### 6.2 Plot area against price

### 6.2. Create an area-price scatter plot

```python
sns.scatterplot(data=df, x="area_sqft", y="price_lakhs")
plt.xlabel("Area (square feet)")
plt.ylabel("Price (lakhs)")
plt.title("Area versus house price")
plt.show()
```

**Expected output**

```text
A scatter plot with a positive relationship
```

**Explanation**

This provides an initial view of one feature's relationship with the target.



### 6.3 Plot distance against price

### 6.3. Create a distance-price scatter plot

```python
sns.scatterplot(data=df, x="distance_km", y="price_lakhs")
plt.xlabel("Distance from city centre (km)")
plt.ylabel("Price (lakhs)")
plt.title("Distance versus house price")
plt.show()
```

**Expected output**

```text
A scatter plot showing a negative pattern in this constructed dataset
```

**Explanation**

The relationship is illustrative. Real property markets may have different patterns and important location effects.



### 6.4 Plot bedrooms against price

### 6.4. Compare price by bedroom count

```python
sns.boxplot(data=df, x="bedrooms", y="price_lakhs")
plt.xlabel("Bedrooms")
plt.ylabel("Price (lakhs)")
plt.title("House price by bedroom count")
plt.show()
```

**Expected output**

```text
Box plots comparing prices across bedroom counts
```

**Explanation**

Group comparisons can reveal patterns, but house size and location may also explain price differences.



## 7. Split the dataset

### 7.1 Import train_test_split

### 7.1. Import split function

```python
from sklearn.model_selection import train_test_split

print("Split tool ready")
```

**Expected output**

```text
Split tool ready
```

**Explanation**

The function creates aligned training and test partitions.



### 7.2 Split X and y

### 7.2. Create training and test sets

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
(16, 4), (4, 4), (16,), (4,)
```

**Explanation**

The model will learn from 16 houses and be evaluated on four held-out houses.



### 7.3 Check feature columns after splitting

### 7.3. Confirm feature order

```python
print(X_train.columns.tolist())
print(X_test.columns.tolist())
```

**Expected output**

```text
The same four feature names in the same order
```

**Explanation**

A model expects the same feature schema during training and prediction.



## 8. Train multiple linear regression

### 8.1 Import LinearRegression

### 8.1. Import the estimator

```python
from sklearn.linear_model import LinearRegression

print("LinearRegression ready")
```

**Expected output**

```text
LinearRegression ready
```

**Explanation**

This estimator can use multiple input columns to predict a numerical target.



### 8.2 Create the model

### 8.2. Initialize the estimator

```python
model = LinearRegression()

print(model)
```

**Expected output**

```text
LinearRegression()
```

**Explanation**

The model is initialized but not fitted yet.



### 8.3 Fit using training data

### 8.3. Train the model

```python
model.fit(X_train, y_train)

print("Model fitted")
```

**Expected output**

```text
Model fitted
```

**Explanation**

The model learns one coefficient for each feature and an intercept using the training partition.



### 8.4 Inspect coefficients

### 8.4. Read learned coefficients

```python
print(model.coef_)
```

**Expected output**

```text
A four-element coefficient array
```

**Explanation**

The coefficient order corresponds to the feature-column order in X_train. The coefficients have different units because the features use different units.



### 8.5 Create a coefficient table

### 8.5. Match coefficients with feature names

```python
coefficient_table = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_
})

print(coefficient_table)
```

**Expected output**

```text
A four-row table pairing each feature with its coefficient
```

**Explanation**

Never report a coefficient without its feature name and units.



### 8.6 Inspect intercept

### 8.6. Read model intercept

```python
print(model.intercept_)
```

**Expected output**

```text
A numerical intercept
```

**Explanation**

The intercept is the model's predicted value when all features are zero. That combination may be outside the realistic range of houses in this dataset.



## 9. Generate predictions

### 9.1 Predict on test observations

### 9.1. Create test predictions

```python
predictions = model.predict(X_test)

print(predictions)
```

**Expected output**

```text
Four predicted house-price values
```

**Explanation**

The model uses all four input features to predict each held-out house price.



### 9.2 Compare actual and predicted prices

### 9.2. Create a comparison table

```python
comparison = X_test.copy()
comparison["actual_price"] = y_test
comparison["predicted_price"] = predictions

print(comparison)
```

**Expected output**

```text
A table with test features, actual price and predicted price
```

**Explanation**

The comparison table helps inspect individual predictions and potential error patterns.



### 9.3 Predict a new house

### 9.3. Create one new property record

```python
new_house = pd.DataFrame({
    "area_sqft": [1250],
    "bedrooms": [3],
    "age_years": [5],
    "distance_km": [4]
})

print(new_house)
```

**Expected output**

```text
A one-row DataFrame with four feature values
```

**Explanation**

The feature names and order match the training schema.



### 9.4 Predict the new property price

### 9.4. Generate a new prediction

```python
new_price = model.predict(new_house)

print(new_price[0])
```

**Expected output**

```text
A numerical predicted price
```

**Explanation**

This prediction is based on the relationships learned from the training sample. Check whether the new house is within the range represented in training.



### 9.5 Predict several properties

### 9.5. Predict multiple new properties

```python
new_houses = pd.DataFrame({
    "area_sqft": [900, 1300, 1800],
    "bedrooms": [2, 3, 4],
    "age_years": [8, 5, 3],
    "distance_km": [8, 5, 2]
})

print(model.predict(new_houses))
```

**Expected output**

```text
Three numerical price predictions
```

**Explanation**

The model predicts each row using the corresponding feature values.



## 10. Evaluate the model

Detailed regression metrics are covered in Day 20. Here we use a quick evaluation to compare actual and predicted values.

### 10.1 Calculate MAE

### 10.1. Calculate mean absolute error

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print(mae)
```

**Expected output**

```text
A non-negative error value in lakhs
```

**Explanation**

MAE is expressed in the same units as the price target.



### 10.2 Calculate R-squared

### 10.2. Calculate R-squared

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)

print(r2)
```

**Expected output**

```text
An R-squared value
```

**Explanation**

Interpret it with the small test sample and other error diagnostics. A high value does not guarantee that all segments are predicted well.



### 10.3 Calculate residuals

### 10.3. Create residual values

```python
residuals = y_test - predictions

print(residuals)
```

**Expected output**

```text
A residual value for each test observation
```

**Explanation**

Positive values indicate underprediction; negative values indicate overprediction.



## 11. Compare simple and multiple regression

A useful experiment compares a model using only area with a model using area, bedrooms, age and distance.

### 11.1 Train a simple model

### 11.1. Create a single-feature model

```python
X_simple = df[["area_sqft"]]

X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(
    X_simple,
    y,
    test_size=0.2,
    random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_simple_train)

print("Simple model fitted")
```

**Expected output**

```text
Simple model fitted
```

**Explanation**

The simple model uses only area. The same split seed creates a comparable test partition for this dataset.



### 11.2 Predict using simple regression

### 11.2. Generate simple model predictions

```python
simple_predictions = simple_model.predict(X_simple_test)

print(simple_predictions)
```

**Expected output**

```text
Four predicted house prices
```

**Explanation**

These predictions use area alone.



### 11.3 Compare MAE values

### 11.3. Compare simple and multiple models

```python
simple_mae = mean_absolute_error(y_simple_test, simple_predictions)
multiple_mae = mean_absolute_error(y_test, predictions)

print("Simple MAE:", simple_mae)
print("Multiple MAE:", multiple_mae)
```

**Expected output**

```text
Two MAE values
```

**Explanation**

The models use the same held-out observations. The comparison shows whether the extra features help on this sample; use cross-validation or more data for a robust conclusion.



## 12. Understand coefficient interpretation

In multiple regression, each coefficient describes the model's predicted change associated with that feature while holding the other included features constant.

Interpret carefully:
- Area is measured in square feet.
- Bedrooms are measured as a count.
- Property age is measured in years.
- Distance is measured in kilometres.
- A coefficient's size depends on the unit and range of its feature.
- Correlated predictors can make coefficients unstable.
- Coefficients do not prove causality.

## 13. Multicollinearity and feature relationships

If two predictors contain overlapping information, their individual coefficients may become difficult to interpret.

Examples:
- Area in square feet and area in square metres.
- Total rooms and bedrooms plus bathrooms.
- Monthly spend and annualized spend.

### 13.1 Inspect predictor correlations

### 13.1. Create a predictor correlation matrix

```python
print(X.corr().round(2))
```

**Expected output**

```text
A correlation matrix for the four predictors
```

**Explanation**

Strong pairwise relationships are a reason to investigate possible redundancy. They do not automatically require feature removal.



### 13.2 Compare coefficient stability

A useful diagnostic is to compare coefficients across sensible feature sets or training folds. If coefficients change substantially when a correlated predictor is added or removed, interpret individual coefficients cautiously.

### 13.3 Do not remove features blindly

Consider:
- Business meaning.
- Prediction-time availability.
- Validation performance.
- Coefficient stability.
- Interpretability requirements.
- Whether features are derived from one another.

## 14. Residual analysis

### 14.1 Create a residual DataFrame

### 14.1. Combine predictions and residuals

```python
residual_df = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions,
    "residual": residuals
})

print(residual_df)
```

**Expected output**

```text
A test-set residual table
```

**Explanation**

Use the same row index for actual and predicted values so that observations remain aligned.



### 14.2 Plot residuals against predictions

### 14.2. Create a residual plot

```python
plt.scatter(predictions, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted price (lakhs)")
plt.ylabel("Residual (lakhs)")
plt.title("Multiple regression residuals")
plt.show()
```

**Expected output**

```text
A residual scatter plot
```

**Explanation**

Look for curvature, clusters or a changing spread. Four test observations are far too few for strong diagnostic conclusions; use more data in practice.



### 14.3 Inspect residuals by price range

### 14.3. Sort residuals for review

```python
print(residual_df.sort_values("actual"))
```

**Expected output**

```text
Rows ordered by actual price
```

**Explanation**

This helps identify whether prediction errors differ for lower- and higher-priced observations.



## 15. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Include target in X | Model receives the answer | Keep target separate |
| Forget feature order | Coefficients apply to the wrong inputs | Preserve named schema |
| Interpret coefficients without units | Numerical size is misleading | State feature and target units |
| Assume extra features always improve the model | Irrelevant features can add noise | Compare using fair validation |
| Ignore correlated predictors | Coefficients may be unstable | Inspect relationships and stability |
| Treat coefficients as causal effects | Regression association is not causation | State limits of interpretation |
| Predict far outside training range | Extrapolation may be unreliable | Validate supported input ranges |
| Ignore residuals | Systematic segment errors may remain hidden | Inspect residual patterns and groups |

## 16. Interview-ready explanations

**What is multiple linear regression?**  
It is a supervised regression model that predicts a numerical target using more than one input feature.

**How is it different from simple linear regression?**  
Simple linear regression uses one feature; multiple linear regression uses several features.

**How do you interpret a coefficient?**  
It describes the model's predicted change associated with a one-unit change in that feature while holding other included features constant. Units, scaling and correlated features affect interpretation.

**Why might adding features not improve the model?**  
Some features may be irrelevant, noisy, redundant or unavailable at prediction time. Compare feature sets using appropriate validation.

**What is multicollinearity?**  
It is a strong linear relationship among predictor features that can make individual coefficients unstable or difficult to interpret.

## 17. Scenario questions

1. A house-price model uses area, bedrooms and number of rooms. What do you inspect?  
   **Answer:** Whether bedrooms and total rooms overlap in meaning and create redundant information.

2. A coefficient for distance is negative. What does it mean?  
   **Answer:** Within this fitted model and data, greater distance is associated with a lower predicted price when other included features are held constant. It does not prove causation.

3. Adding bedrooms reduces training error but increases test error. What might be happening?  
   **Answer:** The model may be overfitting or the feature may not generalize. Compare with validation and inspect feature quality.

4. The model performs well overall but poorly for older properties. What do you do?  
   **Answer:** Compare residuals by property-age group, inspect sample size and consider missing features or different relationships.

5. An API request provides only three of four expected features. What should happen?  
   **Answer:** Reject it with a clear schema-validation error rather than inventing a value silently.

## 18. Lesson completion checklist

- [ ] I can explain multiple linear regression.
- [ ] I can select several features and one numerical target.
- [ ] I can split and train the model.
- [ ] I can inspect coefficients with feature names.
- [ ] I can predict new observations.
- [ ] I can compare simple and multiple regression.
- [ ] I understand multicollinearity and coefficient limitations.
- [ ] I can inspect residuals and segment-level errors.
- [ ] I can explain why additional features do not always improve generalization.

**Next lesson:** Day 23 — Polynomial Regression and Nonlinear Relationships.
