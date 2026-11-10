# Day 24: Regularization — Ridge and Lasso

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain regularization in simple language.
- Understand why very large or unstable coefficients can be a problem.
- Compare ordinary linear regression, Ridge and Lasso.
- Explain the role of alpha.
- Understand coefficient shrinkage.
- Recognize that Lasso can set some coefficients to zero.
- Scale features before regularized regression when units differ.
- Select regularization strength using validation or cross-validation.
- Avoid choosing alpha using the final test set.

## 2. What is regularization?

**Regularization** adds a penalty to the model's training objective to discourage overly large coefficients.

It can help:
- Reduce model complexity.
- Improve stability.
- Reduce overfitting in some situations.
- Handle many or correlated features.
- Improve generalization when the penalty is selected appropriately.

Regularization does not guarantee better performance. The penalty strength must be validated.

## 3. Understand Ridge and Lasso

### Ridge regression

Ridge uses an L2 penalty. It generally shrinks coefficients towards zero but does not usually set them exactly to zero.

It can be useful when:
- Many features contain useful information.
- Predictors are correlated.
- You want to reduce coefficient magnitude while retaining the features.

### Lasso regression

Lasso uses an L1 penalty. It can set some coefficients exactly to zero, producing a sparse coefficient representation.

It can be useful when:
- A smaller set of active features is desired.
- You want a model with some coefficients removed.
- Feature selection is part of the model-development process.

**Important:** A coefficient set to zero in one fitted Lasso model does not prove the feature is universally useless.

## 4. Create a sample regression dataset

### 4.1 Import libraries

### 4.1. Import numerical and ML libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

Pandas and NumPy prepare the dataset. Matplotlib supports coefficient comparisons.



### 4.2 Create a dataset with several features

### 4.2. Create marketing features and sales target

```python
rng = np.random.default_rng(42)

n = 100
df = pd.DataFrame({
    "tv_spend": rng.uniform(10, 100, n),
    "radio_spend": rng.uniform(5, 60, n),
    "online_spend": rng.uniform(2, 40, n),
    "store_visits": rng.integers(10, 100, n)
})

df["sales"] = (
    2.5 * df["tv_spend"]
    + 1.8 * df["radio_spend"]
    + 1.2 * df["online_spend"]
    + 0.3 * df["store_visits"]
    + rng.normal(0, 10, n)
)

print(df.head())
```

**Expected output**

```text
The first five marketing observations
```

**Explanation**

The target is sales. Features have different ranges and units, so scaling is important before regularized regression.



### 4.3 Inspect feature scales

### 4.3. Review summary statistics

```python
print(df.describe().T)
```

**Expected output**

```text
Summary statistics for marketing features and sales
```

**Explanation**

The ranges differ across columns. Coefficient penalties are affected by feature scale, so scale features appropriately.



### 4.4 Separate features and target

### 4.4. Create X and y

```python
X = df.drop(columns=["sales"])
y = df["sales"]

print(X.shape, y.shape)
```

**Expected output**

```text
(100, 4) and (100,)
```

**Explanation**

X contains four predictors. y contains the sales target.



## 5. Split data before preprocessing

### 5.1 Import train_test_split

### 5.1. Import splitting function

```python
from sklearn.model_selection import train_test_split

print("Split tool ready")
```

**Expected output**

```text
Split tool ready
```

**Explanation**

The test set must remain separate while model settings are selected.



### 5.2 Create training and test sets

### 5.2. Split the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
(80, 4) and (20, 4)
```

**Explanation**

Training data is used to fit preprocessing and model parameters. Test data is reserved for final evaluation.



## 6. Fit ordinary Linear Regression as a reference

### 6.1 Import LinearRegression

### 6.1. Import the ordinary regression estimator

```python
from sklearn.linear_model import LinearRegression

print("LinearRegression ready")
```

**Expected output**

```text
LinearRegression ready
```

**Explanation**

This model does not apply Ridge or Lasso regularization.



### 6.2 Train ordinary regression

### 6.2. Fit the reference model

```python
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

print("Reference model fitted")
```

**Expected output**

```text
Reference model fitted
```

**Explanation**

The model learns coefficients without the Ridge or Lasso penalty.



### 6.3 Inspect ordinary coefficients

### 6.3. Create a coefficient table

```python
linear_coefficients = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": linear_model.coef_
})

print(linear_coefficients)
```

**Expected output**

```text
A table containing four feature coefficients
```

**Explanation**

This provides a reference for comparing coefficient changes after regularization.



## 7. Why feature scaling matters for regularization

Ridge and Lasso penalize coefficient sizes. If one feature is measured in thousands and another in fractions, their coefficient magnitudes can differ because of units rather than importance.

Scaling makes the penalty more comparable across numerical features.

### 7.1 Import StandardScaler and Pipeline

### 7.1. Import scaling and pipeline tools

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Scaling tools ready")
```

**Expected output**

```text
Scaling tools ready
```

**Explanation**

A pipeline ensures the scaler is fitted on training data and reused for held-out data.



### 7.2 Create a scaled ordinary regression pipeline

### 7.2. Build a scale and linear model pipeline

```python
scaled_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

print(scaled_linear)
```

**Expected output**

```text
A pipeline containing StandardScaler and LinearRegression
```

**Explanation**

The scaler transforms features before the regression estimator fits them.



### 7.3 Fit the scaled reference model

### 7.3. Train the scaled pipeline

```python
scaled_linear.fit(X_train, y_train)

print("Scaled reference fitted")
```

**Expected output**

```text
Scaled reference fitted
```

**Explanation**

The scaler learns statistics from X_train only.



## 8. Train Ridge regression

### 8.1 Import Ridge

### 8.1. Import Ridge

```python
from sklearn.linear_model import Ridge

print("Ridge imported")
```

**Expected output**

```text
Ridge imported
```

**Explanation**

Ridge applies an L2 penalty to discourage large coefficient values.



### 8.2 Create a Ridge pipeline

### 8.2. Build a Ridge model with scaling

```python
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

print(ridge_pipeline)
```

**Expected output**

```text
A pipeline with StandardScaler and Ridge
```

**Explanation**

alpha controls the regularization strength. The value 1.0 is a starting setting, not a universal best choice.



### 8.3 Fit Ridge

### 8.3. Train the Ridge pipeline

```python
ridge_pipeline.fit(X_train, y_train)

print("Ridge model fitted")
```

**Expected output**

```text
Ridge model fitted
```

**Explanation**

The pipeline scales training features and fits Ridge on the transformed data.



### 8.4 Inspect Ridge coefficients

### 8.4. Read Ridge coefficients

```python
ridge_coefficients = ridge_pipeline.named_steps["model"].coef_

ridge_table = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": ridge_coefficients
})

print(ridge_table)
```

**Expected output**

```text
A coefficient table for the Ridge model
```

**Explanation**

These coefficients correspond to standardized input features. Interpret them differently from coefficients trained on original feature units.



### 8.5 Predict with Ridge

### 8.5. Generate Ridge test predictions

```python
ridge_predictions = ridge_pipeline.predict(X_test)

print(ridge_predictions[:5])
```

**Expected output**

```text
The first five predicted sales values
```

**Explanation**

The pipeline applies the fitted scaler and then the Ridge estimator.



## 9. Train Lasso regression

### 9.1 Import Lasso

### 9.1. Import Lasso

```python
from sklearn.linear_model import Lasso

print("Lasso imported")
```

**Expected output**

```text
Lasso imported
```

**Explanation**

Lasso applies an L1 penalty and can set some coefficients to exactly zero.



### 9.2 Create a Lasso pipeline

### 9.2. Build a Lasso model with scaling

```python
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(alpha=0.1, max_iter=10000))
])

print(lasso_pipeline)
```

**Expected output**

```text
A pipeline with StandardScaler and Lasso
```

**Explanation**

The max_iter value provides an iteration budget. Lasso may need a higher limit for some datasets and settings.



### 9.3 Fit Lasso

### 9.3. Train the Lasso pipeline

```python
lasso_pipeline.fit(X_train, y_train)

print("Lasso model fitted")
```

**Expected output**

```text
Lasso model fitted
```

**Explanation**

The pipeline learns scaling from training data and fits the penalized regression model.



### 9.4 Inspect Lasso coefficients

### 9.4. Create a Lasso coefficient table

```python
lasso_coefficients = lasso_pipeline.named_steps["model"].coef_

lasso_table = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": lasso_coefficients
})

print(lasso_table)
```

**Expected output**

```text
A coefficient table for Lasso
```

**Explanation**

Some coefficients may be exactly zero depending on alpha and the data. A zero coefficient means this fitted model does not use that feature in its linear prediction.



### 9.5 Predict with Lasso

### 9.5. Generate Lasso test predictions

```python
lasso_predictions = lasso_pipeline.predict(X_test)

print(lasso_predictions[:5])
```

**Expected output**

```text
The first five Lasso predictions
```

**Explanation**

The same fitted preprocessing is applied to the held-out data.



## 10. Compare Linear, Ridge and Lasso

### 10.1 Import regression metrics

### 10.1. Import MAE and RMSE

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Regression metrics ready")
```

**Expected output**

```text
Regression metrics ready
```

**Explanation**

Use the same held-out observations for all candidate models.



### 10.2 Calculate errors

### 10.2. Compare model errors

```python
model_predictions = {
    "Linear": scaled_linear.predict(X_test),
    "Ridge": ridge_predictions,
    "Lasso": lasso_predictions
}

comparison_rows = []

for name, prediction in model_predictions.items():
    comparison_rows.append({
        "model": name,
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test, prediction))
    })

comparison = pd.DataFrame(comparison_rows)

print(comparison)
```

**Expected output**

```text
A table with MAE and RMSE for each model
```

**Explanation**

The comparison is based on one split. Use cross-validation to select alpha more reliably.



### 10.3 Compare coefficient magnitudes

### 10.3. Create a coefficient comparison table

```python
coefficient_comparison = pd.DataFrame({
    "feature": X_train.columns,
    "Linear": scaled_linear.named_steps["model"].coef_,
    "Ridge": ridge_coefficients,
    "Lasso": lasso_coefficients
})

print(coefficient_comparison)
```

**Expected output**

```text
A table comparing feature coefficients across three models
```

**Explanation**

The models use standardized features, so coefficient magnitudes are more comparable than if raw features had very different units.



### 10.4 Plot coefficient comparison

### 10.4. Visualize coefficient changes

```python
coefficient_comparison.set_index("feature").plot(kind="bar")
plt.ylabel("Coefficient on standardized feature")
plt.title("Linear, Ridge and Lasso coefficients")
plt.xticks(rotation=30)
plt.show()
```

**Expected output**

```text
A grouped bar chart of coefficients
```

**Explanation**

The chart helps inspect shrinkage and feature removal. It is not a causal feature-importance chart.



## 11. Understand alpha

**alpha** controls regularization strength.

- Smaller alpha applies a weaker penalty.
- Larger alpha applies a stronger penalty.
- Very strong regularization can make the model too simple and underfit.
- Very weak regularization may not control coefficient size enough.

The best value depends on the dataset and must be selected using validation or cross-validation.

### 11.1 Train Ridge with several alpha values

### 11.1. Compare Ridge alpha settings

```python
alpha_values = [0.01, 0.1, 1, 10, 100]

ridge_alpha_results = []

for alpha in alpha_values:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha))
    ])

    candidate.fit(X_train, y_train)
    prediction = candidate.predict(X_test)

    ridge_alpha_results.append({
        "alpha": alpha,
        "MAE": mean_absolute_error(y_test, prediction)
    })

print(pd.DataFrame(ridge_alpha_results))
```

**Expected output**

```text
A table of Ridge alpha values and corresponding MAE
```

**Explanation**

This illustrates how alpha changes model behaviour. Do not choose the best value from repeated test-set comparisons in a real evaluation; use validation or cross-validation.



### 11.2 Train Lasso with several alpha values

### 11.2. Compare Lasso alpha settings

```python
lasso_alpha_results = []

for alpha in [0.001, 0.01, 0.1, 1, 10]:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=alpha, max_iter=10000))
    ])

    candidate.fit(X_train, y_train)
    prediction = candidate.predict(X_test)

    lasso_alpha_results.append({
        "alpha": alpha,
        "MAE": mean_absolute_error(y_test, prediction),
        "nonzero_coefficients": int(
            np.count_nonzero(candidate.named_steps["model"].coef_)
        )
    })

print(pd.DataFrame(lasso_alpha_results))
```

**Expected output**

```text
A table of alpha, MAE and number of non-zero coefficients
```

**Explanation**

As alpha increases, Lasso may set more coefficients to zero. The exact pattern depends on the data and feature scaling.



## 12. Select alpha using cross-validation

### 12.1 Import RidgeCV

### 12.1. Import RidgeCV

```python
from sklearn.linear_model import RidgeCV

print("RidgeCV imported")
```

**Expected output**

```text
RidgeCV imported
```

**Explanation**

RidgeCV evaluates candidate alpha values using cross-validation on the supplied training data.



### 12.2 Create a RidgeCV pipeline

### 12.2. Build a cross-validated Ridge pipeline

```python
ridge_cv_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
])

print(ridge_cv_pipeline)
```

**Expected output**

```text
A pipeline containing StandardScaler and RidgeCV
```

**Explanation**

The pipeline allows the scaler to be fitted within the training workflow. RidgeCV chooses alpha using its internal validation procedure.



### 12.3 Fit RidgeCV

### 12.3. Train RidgeCV on training data

```python
ridge_cv_pipeline.fit(X_train, y_train)

print("RidgeCV fitted")
```

**Expected output**

```text
RidgeCV fitted
```

**Explanation**

Only training data is passed to fit. The held-out test set remains separate.



### 12.4 Inspect selected alpha

### 12.4. Read selected regularization strength

```python
selected_alpha = ridge_cv_pipeline.named_steps["model"].alpha_

print(selected_alpha)
```

**Expected output**

```text
One of the supplied alpha values
```

**Explanation**

The selected alpha is the value chosen by the cross-validation procedure on training data.



### 12.5 Evaluate the selected model once on test data

### 12.5. Generate final test predictions

```python
ridge_cv_predictions = ridge_cv_pipeline.predict(X_test)

print(mean_absolute_error(y_test, ridge_cv_predictions))
```

**Expected output**

```text
A held-out MAE value
```

**Explanation**

Use the test set after model settings have been selected. Do not repeatedly tune based on this final test result.



## 13. Ridge versus Lasso

| Topic | Ridge | Lasso |
|---|---|---|
| Penalty | L2 | L1 |
| Coefficients | Usually shrinks towards zero | Can set coefficients exactly to zero |
| Feature removal | Does not usually remove features entirely | Can produce sparse coefficients |
| Correlated predictors | Often distributes weight across correlated inputs | May select one and suppress others, depending on data |
| Use case | Retain many useful predictors while controlling magnitude | Prefer a sparse coefficient representation |

## 14. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Apply regularization without scaling | Penalty is affected by feature units | Scale within a pipeline |
| Assume Lasso zero means feature is useless | Selection depends on data and alpha | Validate stability and business meaning |
| Choose alpha using test data repeatedly | Test set becomes part of model selection | Use validation or cross-validation |
| Use a very large alpha | Model may underfit | Compare a sensible range |
| Assume Ridge always beats Lasso | Different data structures favour different penalties | Compare both on suitable validation data |
| Compare raw coefficient sizes across different units | Magnitudes are not directly comparable | Consider scaling and units |
| Ignore convergence warnings | The fitted solution may be unreliable | Inspect scaling, alpha and iteration settings |

## 15. Interview-ready explanations

**What is regularization?**  
It adds a penalty to discourage overly large coefficients. It can reduce model complexity and improve generalization when the penalty is selected appropriately.

**What is the difference between Ridge and Lasso?**  
Ridge uses an L2 penalty and generally shrinks coefficients. Lasso uses an L1 penalty and can set some coefficients exactly to zero.

**What is alpha?**  
It controls regularization strength. Larger alpha generally means stronger penalty.

**Why scale features before Ridge or Lasso?**  
The penalty acts on coefficient size, and feature units affect coefficient magnitude. Scaling makes the penalty more comparable across numerical inputs.

**Does Lasso perform feature selection?**  
It can create a sparse coefficient representation by setting some coefficients to zero. The selected set can vary with data, scaling and alpha.

## 16. Scenario questions

1. Coefficients are very large and unstable. What can you try?  
   **Answer:** Check feature scaling, multicollinearity and consider regularization.

2. Lasso sets a feature coefficient to zero. Does that mean the feature is universally useless?  
   **Answer:** No. It means this fitted model with this data and alpha does not use the feature.

3. Should alpha be chosen using the final test score?  
   **Answer:** No. Use validation or cross-validation and reserve the test set for final evaluation.

4. Ridge performs better than Lasso on validation data. What might that suggest?  
   **Answer:** Retaining many predictors with smaller coefficients may work better than a sparse representation for this task.

5. A regularized model performs worse than ordinary regression. What do you investigate?  
   **Answer:** Alpha strength, feature scaling, data size, validation stability and whether regularization is actually needed.

## 17. Lesson completion checklist

- [ ] I can explain regularization.
- [ ] I can distinguish Ridge and Lasso.
- [ ] I understand alpha.
- [ ] I can scale features inside a pipeline.
- [ ] I can train Ridge and Lasso models.
- [ ] I can compare coefficient magnitudes.
- [ ] I can inspect Lasso zero coefficients.
- [ ] I can select alpha using cross-validation.
- [ ] I can explain why test data should not guide repeated tuning.

**Next lesson:** Day 25 — Logistic Regression and Sigmoid Function.
