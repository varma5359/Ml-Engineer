# Day 23: Polynomial Regression and Nonlinear Relationships

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain why a straight-line model may not represent every relationship.
- Recognize linear and nonlinear patterns using charts and residuals.
- Explain polynomial feature transformation in simple language.
- Create squared and higher-order feature terms using scikit-learn.
- Train polynomial regression using a Pipeline.
- Compare different polynomial degrees.
- Understand how high-degree models can overfit.
- Validate polynomial models using held-out data or cross-validation.
- Recognize why extrapolation outside the training range can be unreliable.

## 2. Why might a straight-line model be insufficient?

A linear regression model represents a straight-line relationship in a one-feature problem. Some business relationships are curved.

Examples:
- Advertising may increase sales rapidly at first, then show diminishing returns.
- Energy consumption may increase at both very low and very high temperatures.
- Vehicle fuel efficiency may change nonlinearly with speed.
- A product's demand may increase and later decrease as price changes.

When a scatter plot or residual plot shows a curved pattern, a straight-line model may be too simple.

## 3. What is polynomial regression?

**Polynomial regression** creates transformed features such as squared or cubed versions of existing features, then fits a regression model using those transformed inputs.

The model is still linear in its learned coefficients, but the transformed features allow it to represent curved patterns in the original feature space.

You do not need to memorize an equation. Understand the workflow:
1. Start with an original feature.
2. Generate additional polynomial feature columns.
3. Fit a regression estimator on the expanded feature table.
4. Compare performance and residuals.
5. Choose model complexity using validation data.

## 4. Create a curved dataset

### 4.1 Import libraries

### 4.1. Import Pandas, NumPy and plotting tools

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

NumPy creates numerical examples, Pandas stores the data and plotting libraries visualize the relationship.



### 4.2 Generate a curved relationship

### 4.2. Create advertising and sales data

```python
rng = np.random.default_rng(42)

advertising = np.linspace(1, 20, 40)
sales = 5 + 3 * advertising - 0.08 * advertising ** 2
sales = sales + rng.normal(0, 2, size=len(advertising))

df = pd.DataFrame({
    "advertising_spend": advertising,
    "sales": sales
})

print(df.head())
```

**Expected output**

```text
The first five advertising and sales rows
```

**Explanation**

The example includes a curved trend and random noise. It is simulated for learning, not a claim about a real advertising market.



### 4.3 Inspect the dataset

### 4.3. Check the shape and summary

```python
print(df.shape)
print(df.describe())
```

**Expected output**

```text
(40, 2) and summary statistics
```

**Explanation**

There are 40 observations and two columns. Inspect the range before choosing a model.



### 4.4 Plot the relationship

### 4.4. Create a scatter plot

```python
sns.scatterplot(data=df, x="advertising_spend", y="sales")
plt.xlabel("Advertising spend")
plt.ylabel("Sales")
plt.title("Advertising spend versus sales")
plt.show()
```

**Expected output**

```text
A scatter plot showing a curved upward pattern
```

**Explanation**

The chart suggests the relationship may not be represented well by a straight line.



## 5. Fit a simple linear regression baseline

### 5.1 Select features and target

### 5.1. Create X and y

```python
X = df[["advertising_spend"]]
y = df["sales"]

print(X.shape, y.shape)
```

**Expected output**

```text
(40, 1) and (40,)
```

**Explanation**

The feature is advertising spend and the target is sales.



### 5.2 Split data

### 5.2. Create training and test partitions

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
(30, 1) and (10, 1)
```

**Explanation**

The test observations are held back for comparison. The split is random because this is a simulated independent-observation example.



### 5.3 Train simple linear regression

### 5.3. Fit a straight-line model

```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

print("Linear model fitted")
```

**Expected output**

```text
Linear model fitted
```

**Explanation**

The model learns a straight-line relationship from training observations.



### 5.4 Predict test values

### 5.4. Generate linear predictions

```python
linear_predictions = linear_model.predict(X_test)

print(linear_predictions[:5])
```

**Expected output**

```text
The first five predicted sales values
```

**Explanation**

These predictions provide a baseline for comparing a more flexible model.



### 5.5 Calculate baseline error

### 5.5. Calculate baseline MAE

```python
from sklearn.metrics import mean_absolute_error

linear_mae = mean_absolute_error(y_test, linear_predictions)

print(linear_mae)
```

**Expected output**

```text
A non-negative MAE value
```

**Explanation**

MAE summarizes the average absolute error on the held-out observations.



### 5.6 Plot the fitted straight line

### 5.6. Visualize the linear fit

```python
plot_x = pd.DataFrame({
    "advertising_spend": np.linspace(
        X["advertising_spend"].min(),
        X["advertising_spend"].max(),
        100
    )
})

plot_y = linear_model.predict(plot_x)

plt.scatter(X_train["advertising_spend"], y_train, label="Training data")
plt.plot(plot_x["advertising_spend"], plot_y, label="Linear fit")
plt.xlabel("Advertising spend")
plt.ylabel("Sales")
plt.title("Simple linear regression fit")
plt.legend()
plt.show()
```

**Expected output**

```text
A straight fitted line over the training observations
```

**Explanation**

The line may miss the curved pattern. This motivates testing transformed features.



## 6. Generate polynomial features

### 6.1 Import PolynomialFeatures

### 6.1. Import the transformer

```python
from sklearn.preprocessing import PolynomialFeatures

print("PolynomialFeatures imported")
```

**Expected output**

```text
PolynomialFeatures imported
```

**Explanation**

This transformer generates additional columns based on combinations and powers of the original features.



### 6.2 Create a degree-two transformer

### 6.2. Initialize degree two

```python
poly2 = PolynomialFeatures(
    degree=2,
    include_bias=False
)

print(poly2)
```

**Expected output**

```text
A PolynomialFeatures configuration
```

**Explanation**

Degree two generates the original feature and its squared term. include_bias=False avoids adding a constant column because LinearRegression already fits an intercept by default.



### 6.3 Fit and transform the training feature

### 6.3. Create degree-two training features

```python
X_train_poly2 = poly2.fit_transform(X_train)

print(X_train_poly2[:5])
```

**Expected output**

```text
A two-column array containing original and squared feature values
```

**Explanation**

The transformer learns the input structure and expands it. Only training data is used for fitting the transformation object.



### 6.4 Transform the test feature

### 6.4. Apply the fitted polynomial transformer

```python
X_test_poly2 = poly2.transform(X_test)

print(X_test_poly2[:5])
```

**Expected output**

```text
A two-column array for test observations
```

**Explanation**

The test data is transformed using the feature structure learned from training. Do not fit a separate polynomial transformer on test data.



### 6.5 Inspect output feature names

### 6.5. Read polynomial feature names

```python
print(poly2.get_feature_names_out(["advertising_spend"]))
```

**Expected output**

```text
Names for the original feature and its squared term
```

**Explanation**

Feature names make the expanded representation easier to understand.



## 7. Train polynomial regression using a Pipeline

A Pipeline applies polynomial transformation and regression in sequence.

### 7.1 Create the pipeline

### 7.1. Connect polynomial features and regression

```python
from sklearn.pipeline import Pipeline

poly2_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("regression", LinearRegression())
])

print(poly2_model)
```

**Expected output**

```text
A pipeline containing PolynomialFeatures and LinearRegression
```

**Explanation**

The pipeline keeps feature generation and model fitting together.



### 7.2 Fit the pipeline

### 7.2. Train degree-two regression

```python
poly2_model.fit(X_train, y_train)

print("Polynomial model fitted")
```

**Expected output**

```text
Polynomial model fitted
```

**Explanation**

The pipeline learns the polynomial feature structure and regression parameters using training data.



### 7.3 Predict test observations

### 7.3. Generate polynomial predictions

```python
poly2_predictions = poly2_model.predict(X_test)

print(poly2_predictions[:5])
```

**Expected output**

```text
The first five polynomial model predictions
```

**Explanation**

The pipeline transforms X_test and applies the fitted regression model.



### 7.4 Evaluate degree-two MAE

### 7.4. Calculate degree-two error

```python
poly2_mae = mean_absolute_error(y_test, poly2_predictions)

print(poly2_mae)
```

**Expected output**

```text
A non-negative MAE value
```

**Explanation**

Compare it with the linear baseline using the same test observations.



### 7.5 Compare linear and polynomial models

### 7.5. Create a model comparison

```python
comparison = pd.DataFrame({
    "model": ["Linear", "Polynomial degree 2"],
    "MAE": [linear_mae, poly2_mae]
})

print(comparison)
```

**Expected output**

```text
A two-row MAE comparison table
```

**Explanation**

A lower value indicates lower average absolute error on this test split. A single split is not enough for a strong generalization claim.



## 8. Compare polynomial degrees

A higher degree creates a more flexible model. Flexibility can help capture a curved relationship, but it can also fit random noise.

### 8.1 Train a degree-three model

### 8.1. Create and fit degree three

```python
poly3_model = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("regression", LinearRegression())
])

poly3_model.fit(X_train, y_train)

print("Degree 3 fitted")
```

**Expected output**

```text
Degree 3 fitted
```

**Explanation**

Degree three creates additional transformed features. The model becomes more flexible.



### 8.2 Evaluate degree three

### 8.2. Predict and calculate error

```python
poly3_predictions = poly3_model.predict(X_test)
poly3_mae = mean_absolute_error(y_test, poly3_predictions)

print(poly3_mae)
```

**Expected output**

```text
A non-negative MAE value
```

**Explanation**

Compare this value with the degree-two and linear models. Do not assume a higher degree must perform better.



### 8.3 Train a high-degree model

### 8.3. Create a degree-twelve model

```python
poly12_model = Pipeline([
    ("poly", PolynomialFeatures(degree=12, include_bias=False)),
    ("regression", LinearRegression())
])

poly12_model.fit(X_train, y_train)

print("Degree 12 fitted")
```

**Expected output**

```text
Degree 12 fitted
```

**Explanation**

A high-degree model can be very flexible. It may fit the training data closely while behaving poorly on new observations.



### 8.4 Compare train and test error

### 8.4. Calculate errors across degrees

```python
models = {
    "Linear": linear_model,
    "Polynomial 2": poly2_model,
    "Polynomial 3": poly3_model,
    "Polynomial 12": poly12_model
}

rows = []

for name, fitted_model in models.items():
    train_pred = fitted_model.predict(X_train)
    test_pred = fitted_model.predict(X_test)

    rows.append({
        "model": name,
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred)
    })

degree_comparison = pd.DataFrame(rows)

print(degree_comparison)
```

**Expected output**

```text
A comparison table with training and test MAE for each model
```

**Explanation**

A large gap between training and test error can indicate overfitting. The exact results depend on the sample and random split.



## 9. Visualize polynomial fits

### 9.1 Create a plotting range

### 9.1. Generate evenly spaced feature values

```python
plot_range = pd.DataFrame({
    "advertising_spend": np.linspace(
        X["advertising_spend"].min(),
        X["advertising_spend"].max(),
        200
    )
})

print(plot_range.head())
```

**Expected output**

```text
The first five plotting feature values
```

**Explanation**

The plotting range covers the observed feature interval. Avoid interpreting far-outside extrapolations as reliable.



### 9.2 Plot linear and polynomial fits

### 9.2. Compare fitted curves

```python
plt.scatter(X["advertising_spend"], y, label="Observed data")

for name, fitted_model in [
    ("Linear", linear_model),
    ("Polynomial 2", poly2_model),
    ("Polynomial 3", poly3_model)
]:
    plt.plot(
        plot_range["advertising_spend"],
        fitted_model.predict(plot_range),
        label=name
    )

plt.xlabel("Advertising spend")
plt.ylabel("Sales")
plt.title("Linear and polynomial regression comparison")
plt.legend()
plt.show()
```

**Expected output**

```text
A scatter plot with straight and curved fitted relationships
```

**Explanation**

The chart helps compare how each model represents the pattern. Use validation performance and residuals, not visual smoothness alone, to choose a model.



## 10. Cross-validation for degree selection

A single train-test split can produce a noisy comparison. Cross-validation evaluates candidate settings across several training/validation folds.

### 10.1 Import cross-validation tools

### 10.1. Import cross_val_score

```python
from sklearn.model_selection import cross_val_score

print("Cross-validation ready")
```

**Expected output**

```text
Cross-validation ready
```

**Explanation**

Cross-validation helps compare model settings using multiple data partitions.



### 10.2 Evaluate several degrees

### 10.2. Compare degrees using cross-validation

```python
cv_results = []

for degree in [1, 2, 3, 4, 5]:
    candidate = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("regression", LinearRegression())
    ])

    scores = cross_val_score(
        candidate,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    cv_results.append({
        "degree": degree,
        "mean_mae": -scores.mean(),
        "std_mae": scores.std()
    })

cv_table = pd.DataFrame(cv_results)

print(cv_table)
```

**Expected output**

```text
A table showing cross-validated MAE and fold variation for each degree
```

**Explanation**

Scikit-learn returns negative MAE because its scoring convention assumes larger scores are better. Negating the values makes them easier to read as positive error.



### 10.3 Plot validation error by degree

### 10.3. Visualize degree comparison

```python
plt.plot(cv_table["degree"], cv_table["mean_mae"], marker="o")
plt.xlabel("Polynomial degree")
plt.ylabel("Mean cross-validation MAE")
plt.title("Polynomial degree selection")
plt.show()
```

**Expected output**

```text
A line chart comparing average validation MAE across degrees
```

**Explanation**

Choose complexity based on validation performance, stability and business requirements. Do not select a degree based on training fit alone.



## 11. Why high polynomial degrees can overfit

Overfitting occurs when a model learns noise or peculiarities of the training sample instead of patterns that generalize.

Warning signs:
- Very low training error.
- Much higher validation/test error.
- A highly wavy fitted curve.
- Unstable predictions near the edge of the observed feature range.
- Large changes when the training sample changes.

## 12. Limitations and cautions

- Polynomial features can increase the number of columns quickly.
- High-degree models can become numerically unstable.
- Predictions outside the training range can be unreliable.
- A polynomial relationship may not reflect the real business mechanism.
- A more complex model should beat a simple baseline on suitable validation data.
- Feature scaling may be useful for numerical stability in some workflows.

## 13. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Assume higher degree is always better | Compare validation performance |
| Fit polynomial transformer before splitting | Fit it within a pipeline on training data |
| Select degree from test score repeatedly | Use cross-validation or validation data |
| Extrapolate far beyond observed range | Validate supported range and use domain knowledge |
| Interpret transformed terms as original business measures | Explain that they are generated features |
| Ignore training-test gap | Inspect overfitting |
| Use a curve only because it looks smooth | Confirm predictive value and residual behaviour |

## 14. Interview-ready explanations

**What is polynomial regression?**  
It expands original features into polynomial terms such as squared or cubed features and fits a regression model on those transformed columns. This allows curved relationships while remaining linear in learned coefficients.

**Why might a straight-line model fail?**  
The relationship between the feature and target may be nonlinear, so a straight line cannot capture the pattern.

**What happens when polynomial degree is too high?**  
The model can fit noise, become unstable and perform poorly on unseen data.

**How do you select polynomial degree?**  
Use validation or cross-validation, compare error and stability, inspect residuals and avoid selecting complexity based only on training performance.

## 15. Scenario questions

1. A linear model shows a curved residual pattern. What could you test?  
   **Answer:** Polynomial features or another nonlinear model, then validate the result.

2. A degree-12 model fits training data almost perfectly but fails validation. What is likely?  
   **Answer:** Overfitting.

3. Is polynomial regression safe for predictions far beyond the training range?  
   **Answer:** Often not. Polynomial curves can behave unpredictably outside the observed range.

4. Degree two improves validation error slightly but degree five improves training error greatly. Which do you prefer?  
   **Answer:** Prefer the simpler model if its validation performance is comparable and more stable.

5. Why use a Pipeline with PolynomialFeatures?  
   **Answer:** It keeps feature generation and model fitting together and ensures transformations are learned within the training workflow.

## 16. Lesson completion checklist

- [ ] I can explain polynomial regression.
- [ ] I understand why polynomial features create curved relationships.
- [ ] I can use PolynomialFeatures.
- [ ] I can create a polynomial regression Pipeline.
- [ ] I can compare different degrees.
- [ ] I understand overfitting.
- [ ] I can use cross-validation for degree selection.
- [ ] I understand extrapolation limitations.

**Next lesson:** Day 24 — Regularization: Ridge and Lasso.
