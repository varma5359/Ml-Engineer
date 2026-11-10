# Day 21: Gradient Descent and Learning Rate

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain gradient descent in plain language.
- Understand why an optimizer is used during model training.
- Explain the role of a learning rate.
- Recognize a learning rate that is too small or too large.
- Understand the difference between batch, stochastic and mini-batch updates conceptually.
- Train an SGDRegressor using scikit-learn.
- Scale features where appropriate for gradient-based optimization.
- Inspect convergence and model performance.
- Recognize common optimization problems without memorizing mathematical formulas.

## 2. What is gradient descent?

**Gradient descent** is an optimization method used by many machine-learning algorithms to adjust model parameters so that the selected training objective becomes smaller.

Imagine trying to reduce prediction errors:
1. The model begins with initial parameter values.
2. It makes predictions.
3. The training process measures how wrong those predictions are.
4. The optimizer determines how the parameters should change to reduce the error.
5. The process repeats.
6. Training stops when the configured stopping condition is met.

The important idea is iterative improvement. The optimizer updates parameters based on the error signal.

## 3. Why do we need an optimizer?

Some models learn parameters by solving a calculation directly. Other models use iterative optimization.

Gradient-based optimization is common in:
- Linear models trained with stochastic gradient methods.
- Logistic regression implementations.
- Neural networks.
- Deep learning systems.

The optimizer is not the model itself. It is the method used to update parameters during training.

## 4. What is learning rate?

The **learning rate** controls how large a parameter update is during optimization.

Beginner analogy:
- A very small step may move towards the goal slowly.
- A very large step may jump past a useful point and fail to settle.
- A suitable step size helps the training process make steady progress.

| Learning-rate behaviour | Possible effect |
|---|---|
| Too small | Training may be very slow |
| Too large | Loss may oscillate or fail to converge |
| Suitable | Training may make stable progress |
| Changes over time | Can help optimization progress at different stages |

There is no universally best learning rate. It depends on the model, data scale, optimizer and training setup.

## 5. Prepare a small regression dataset

### 5.1 Import libraries

### 5.1. Import required libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Optimization tools ready")
```

**Expected output**

```text
Optimization tools ready
```

**Explanation**

NumPy and Pandas prepare data. Matplotlib visualizes the training behaviour.



### 5.2 Create a simple dataset

### 5.2. Create area and house-price observations

```python
df = pd.DataFrame({
    "area": [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
             1500, 1600, 1700, 1800, 1900, 2000],
    "price": [18, 20, 22, 25, 27, 30, 32, 35, 37, 40,
              42, 45, 47, 50, 52, 55]
})

print(df.head())
```

**Expected output**

```text
The first five area and price observations
```

**Explanation**

The target increases with area in this teaching dataset. Real-world data contains noise and may not follow a perfectly smooth relationship.



### 5.3 Inspect feature ranges

### 5.3. Check summary statistics

```python
print(df.describe())
```

**Expected output**

```text
Summary statistics for area and price
```

**Explanation**

Feature ranges matter for gradient-based optimization. Large differences in scale can make optimization harder for some models.



### 5.4 Separate feature and target

### 5.4. Create X and y

```python
X = df[["area"]]
y = df["price"]

print(X.shape, y.shape)
```

**Expected output**

```text
(16, 1) and (16,)
```

**Explanation**

X is the two-dimensional feature table and y is the numerical target.



## 6. Understand feature scaling before optimization

A feature with values in thousands and another feature with values between zero and one may affect optimization differently. Scaling can help put numerical features on a more comparable range.

### 6.1 Split data before scaling

### 6.1. Create training and test sets

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
Training and test shapes
```

**Explanation**

Split first so that scaling statistics are learned from training data only.



### 6.2 Create a scaler

### 6.2. Initialize StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print(scaler)
```

**Expected output**

```text
StandardScaler()
```

**Explanation**

The scaler object is created but has not learned the training feature statistics yet.



### 6.3 Fit scaler on training features

### 6.3. Fit and transform training data

```python
X_train_scaled = scaler.fit_transform(X_train)

print(X_train_scaled[:5])
```

**Expected output**

```text
The first five standardized training feature values
```

**Explanation**

The scaler learns its parameters from X_train and transforms those rows.



### 6.4 Transform test features

### 6.4. Transform test data with the fitted scaler

```python
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled[:5])
```

**Expected output**

```text
The first five standardized test feature values
```

**Explanation**

Do not fit the scaler again on test data. Apply the training-fitted transformation.



## 7. Train using SGDRegressor

**SGDRegressor** is a scikit-learn regression estimator trained using stochastic gradient descent. It provides a practical way to explore iterative optimization.

### 7.1 Import SGDRegressor

### 7.1. Import the estimator

```python
from sklearn.linear_model import SGDRegressor

print("SGDRegressor imported")
```

**Expected output**

```text
SGDRegressor imported
```

**Explanation**

The estimator supports different learning-rate schedules and regularization settings.



### 7.2 Create a model with a learning rate

### 7.2. Initialize SGDRegressor

```python
sgd_model = SGDRegressor(
    learning_rate="constant",
    eta0=0.01,
    max_iter=2000,
    tol=1e-4,
    random_state=42
)

print(sgd_model)
```

**Expected output**

```text
An SGDRegressor configuration
```

**Explanation**

eta0 is the initial learning-rate value. max_iter sets the iteration limit, tol controls a stopping tolerance and random_state supports repeatability.



### 7.3 Fit the model

### 7.3. Train the SGD model

```python
sgd_model.fit(X_train_scaled, y_train)

print("SGD training completed")
```

**Expected output**

```text
SGD training completed
```

**Explanation**

The estimator iteratively updates its parameters using the training data.



### 7.4 Inspect learned coefficient

### 7.4. Read the learned coefficient

```python
print(sgd_model.coef_)
```

**Expected output**

```text
A one-element coefficient array
```

**Explanation**

The coefficient corresponds to the scaled area feature because the model was trained on scaled input.



### 7.5 Inspect intercept

### 7.5. Read the intercept

```python
print(sgd_model.intercept_)
```

**Expected output**

```text
A numerical intercept
```

**Explanation**

The intercept is part of the fitted prediction function.



### 7.6 Generate test predictions

### 7.6. Predict with the SGD model

```python
sgd_predictions = sgd_model.predict(X_test_scaled)

print(sgd_predictions)
```

**Expected output**

```text
Predicted house-price values for the test observations
```

**Explanation**

The same fitted scaler must be used to transform future data before passing it to the model.



## 8. Compare different learning rates

A learning rate that is too small may need many iterations. A rate that is too large may cause unstable training.

### 8.1 Train with a smaller learning rate

### 8.1. Create a small learning-rate model

```python
small_lr_model = SGDRegressor(
    learning_rate="constant",
    eta0=0.0001,
    max_iter=3000,
    tol=1e-5,
    random_state=42
)

small_lr_model.fit(X_train_scaled, y_train)

print(small_lr_model.coef_)
```

**Expected output**

```text
A learned coefficient array
```

**Explanation**

A small learning rate makes updates smaller. It may require more iterations to reach a useful solution.



### 8.2 Train with a larger learning rate

### 8.2. Create a larger learning-rate model

```python
large_lr_model = SGDRegressor(
    learning_rate="constant",
    eta0=0.1,
    max_iter=3000,
    tol=1e-5,
    random_state=42
)

large_lr_model.fit(X_train_scaled, y_train)

print(large_lr_model.coef_)
```

**Expected output**

```text
A learned coefficient array
```

**Explanation**

A larger learning rate changes parameters more aggressively. It may converge quickly, oscillate or become unstable depending on the data and model.



### 8.3 Compare prediction errors

### 8.3. Calculate MAE for learning-rate choices

```python
from sklearn.metrics import mean_absolute_error

small_pred = small_lr_model.predict(X_test_scaled)
large_pred = large_lr_model.predict(X_test_scaled)

print("Small LR MAE:", mean_absolute_error(y_test, small_pred))
print("Large LR MAE:", mean_absolute_error(y_test, large_pred))
```

**Expected output**

```text
Two MAE values for the learning-rate settings
```

**Explanation**

Compare using the same held-out observations. This is a demonstration; a single split and small sample are not enough for robust hyperparameter selection.



## 9. What happens when learning rate is too small or too large?

### Too small
- Parameter updates are tiny.
- Training can take a long time.
- The model may not reach a useful solution within the iteration budget.

### Too large
- Updates may overshoot a useful region.
- Training loss may oscillate.
- Optimization may diverge or fail to converge.

### Suitable
- Updates make useful progress.
- Training can settle within the configured stopping criteria.
- Validation performance is acceptable.

**Developer habit:** Do not choose a learning rate only because the training code runs. Inspect convergence, warnings and validation performance.

## 10. Batch, stochastic and mini-batch gradient descent

These terms describe how many training observations contribute to an update.

| Approach | Update uses | Beginner explanation |
|---|---|---|
| Batch gradient descent | All training observations | Update after considering the full training dataset |
| Stochastic gradient descent | One observation at a time | Update using one observation |
| Mini-batch gradient descent | A small group of observations | Update using a small batch |

The trade-offs include update frequency, noise, memory use and computational efficiency.

### 10.1 Understand a batch of rows

### 10.1. Select the complete training feature set

```python
print(X_train_scaled.shape)
```

**Expected output**

```text
A shape showing all training observations and one feature
```

**Explanation**

Batch processing uses the complete training set for the update calculation.



### 10.2 Select one observation

### 10.2. Inspect one training row

```python
one_row = X_train_scaled[0:1]

print(one_row)
```

**Expected output**

```text
A one-row array
```

**Explanation**

A stochastic update uses one observation at a time. The SGDRegressor implementation handles this internally.



### 10.3 Select a mini-batch

### 10.3. Inspect a small batch

```python
mini_batch = X_train_scaled[:4]

print(mini_batch.shape)
```

**Expected output**

```text
A shape with four rows and one feature
```

**Explanation**

A mini-batch contains a small group of observations. Many neural-network frameworks expose batch_size directly.



## 11. Optimization and convergence

**Convergence** means the training process has reached a point where further updates produce little change according to the configured stopping rule or objective.

A model may stop because:
- It reaches the maximum iteration count.
- The improvement becomes smaller than a configured tolerance.
- Another stopping condition is met.

A convergence warning should be investigated rather than ignored.

### 11.1 Inspect iteration count

### 11.1. Read the number of iterations

```python
print(sgd_model.n_iter_)
```

**Expected output**

```text
An integer iteration count
```

**Explanation**

This reports the number of iterations used by the fitted estimator.



### 11.2 Inspect training warnings

If scikit-learn reports that the estimator did not converge, check:
- Feature scales.
- Learning rate.
- Maximum iterations.
- Tolerance.
- Data quality.
- Model settings.

Do not simply increase max_iter without understanding the warning.

## 12. Use a pipeline for scaling and SGD

A Pipeline keeps the scaler and estimator together.

### 12.1 Build the pipeline

### 12.1. Create a StandardScaler and SGDRegressor pipeline

```python
from sklearn.pipeline import Pipeline

sgd_pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", SGDRegressor(
        learning_rate="constant",
        eta0=0.01,
        max_iter=2000,
        random_state=42
    ))
])

print(sgd_pipeline)
```

**Expected output**

```text
A Pipeline containing StandardScaler and SGDRegressor
```

**Explanation**

The pipeline learns scaling from training data and applies it consistently when predicting.



### 12.2 Fit the pipeline on raw training features

### 12.2. Train the pipeline

```python
sgd_pipeline.fit(X_train, y_train)

print("Pipeline fitted")
```

**Expected output**

```text
Pipeline fitted
```

**Explanation**

The pipeline handles scaling internally. You pass the original unscaled DataFrame to fit.



### 12.3 Predict using raw test features

### 12.3. Predict with the pipeline

```python
pipeline_predictions = sgd_pipeline.predict(X_test)

print(pipeline_predictions)
```

**Expected output**

```text
Predictions for the test observations
```

**Explanation**

The pipeline transforms X_test using the scaler fitted on X_train, then predicts.



## 13. Common optimization problems

| Problem | Possible reason | What to inspect |
|---|---|---|
| Training is very slow | Learning rate too small or poor feature scale | Scaling, learning-rate schedule and iteration count |
| Training is unstable | Learning rate too large | Convergence warnings and loss behaviour |
| Model does not converge | Iteration limit or settings unsuitable | Scale, max_iter, tol and data quality |
| Different runs produce different results | Random initialization or data order | Seeds and deterministic settings |
| Training error low but validation error high | Overfitting or data mismatch | Split, regularization and data distribution |
| Model predictions are extreme | Feature scale or model settings | Input range and learned coefficients |

## 14. Interview-ready explanations

**What is gradient descent?**  
It is an iterative optimization method that adjusts model parameters to reduce a selected training objective.

**What is learning rate?**  
It controls the size of parameter updates. Too small can make training slow; too large can cause unstable or overshooting updates.

**What is stochastic gradient descent?**  
It updates model parameters using one training observation at a time, rather than the complete training dataset for every update.

**Why scale features for gradient-based optimization?**  
Features with very different numerical ranges can make optimization inefficient or unstable. Scaling can help put inputs on a more comparable range.

**Does a convergence warning mean the model is unusable?**  
Not automatically, but it means the training process should be investigated. Check scaling, learning rate, iteration settings and data quality.

## 15. Scenario questions

1. Training is very slow with a tiny learning rate. What do you investigate?  
   **Answer:** Check feature scaling, learning-rate schedule, iteration count and convergence behaviour.

2. Training loss oscillates with a high learning rate. What might help?  
   **Answer:** Try a smaller or adaptive learning rate and validate the change.

3. A model converges on training but performs poorly on test. Is that an optimizer problem only?  
   **Answer:** No. It may be overfitting, data mismatch, leakage or an unsuitable model.

4. Why should a scaler be fitted on training data only?  
   **Answer:** To prevent held-out information from influencing preprocessing.

5. Does a fixed random_state guarantee identical results across every hardware and library version?  
   **Answer:** No. It improves repeatability but does not control every source of nondeterminism.

## 16. Lesson completion checklist

- [ ] I can explain gradient descent without a formula.
- [ ] I can explain learning rate.
- [ ] I understand small and large learning-rate behaviour.
- [ ] I can describe batch, stochastic and mini-batch updates.
- [ ] I can train an SGDRegressor.
- [ ] I understand why feature scaling matters.
- [ ] I can build a pipeline with scaling and SGD.
- [ ] I can investigate convergence warnings.
- [ ] I can compare settings on held-out data.

**Next lesson:** Day 22 — Multiple Linear Regression.
