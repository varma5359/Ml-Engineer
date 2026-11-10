# Day 52: Model Bias, Variance and Overfitting

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives
- Explain underfitting, overfitting, bias and variance in simple language.
- Compare training and validation performance.
- Recognize signs of overfitting and underfitting.
- Understand how model complexity affects generalization.
- Use learning curves to inspect model behaviour.
- Apply practical methods to reduce overfitting.
- Explain why good training performance is not enough.
- Answer scenario-based interview questions about model generalization.

## 2. What is generalization?

A machine learning model learns patterns from training data. The real goal is to perform well on new observations that were not used during training.

**Generalization** means the model can apply useful learned patterns to unseen data.

A model may:
- Learn too little and miss important patterns.
- Learn the training examples too closely, including noise.
- Learn useful patterns that continue to work on unseen data.

We evaluate generalization using validation and test data that were not used to fit the model.

## 3. Understand underfitting

**Underfitting** happens when a model is too simple or has not learned enough from the available data.

Common signs:
- Poor training performance.
- Poor validation performance.
- Important patterns are not captured.
- Model predictions are too simple for the problem.

Possible reasons:
- Model is too simple.
- Important features are missing.
- Training has not converged.
- Regularization is too strong.
- Data quality is poor.

Possible actions:
- Try a more suitable model.
- Add useful features.
- Reduce excessive regularization.
- Improve data quality.
- Check whether the model has trained sufficiently.

## 4. Understand overfitting

**Overfitting** happens when a model learns the training data too closely, including noise or accidental patterns that do not generalize.

Common signs:
- Very strong training performance.
- Much weaker validation or test performance.
- Predictions change substantially across samples.
- Model is overly complex for the available data.

Possible reasons:
- Model complexity is too high.
- Dataset is too small.
- Too many noisy or irrelevant features.
- Data leakage creates unrealistic training results.
- Hyperparameters are selected too aggressively.

Possible actions:
- Simplify the model.
- Use regularization.
- Collect more representative data.
- Apply feature selection carefully.
- Use cross-validation.
- Use early stopping for iterative models.
- Remove leakage and improve validation design.

## 5. Understand bias and variance

### Bias
Bias is error caused by overly simple assumptions or a model that cannot capture the relevant patterns.

Practical signs:
- The model performs poorly on training data.
- It consistently misses important relationships.
- A more flexible model may be needed.

### Variance
Variance describes how much a model's learned behaviour changes when the training data changes.

Practical signs:
- Training performance is strong but validation performance is weaker.
- Different training samples produce noticeably different models.
- The model may be too sensitive to noise.

### Compare bias and variance

| Concept | Practical meaning | Typical symptom |
|---|---|---|
| High bias | Model is too simple to learn important patterns | Poor training and validation performance |
| High variance | Model is too sensitive to training examples | Strong training, weaker validation performance |
| Underfitting | Model has not captured enough useful structure | Poor training and validation results |
| Overfitting | Model has learned training-specific noise or details | Large training-validation gap |

## 6. Create a regression dataset for demonstration

### 6.1 Import libraries

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Libraries ready")
~~~

### 6.2 Create a nonlinear dataset

~~~python
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

X, y = make_friedman1(
    n_samples=500,
    n_features=10,
    noise=1.5,
    random_state=42
)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
~~~

**Expected output**
~~~text
Feature shape: (500, 10)
Target shape: (500,)
~~~

### 6.3 Split training and validation data

~~~python
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print(X_train.shape, X_valid.shape)
~~~

**Explanation**
We use a validation subset to compare model behaviour. A final test set should remain untouched for final reporting in a complete project.

## 7. Compare a simple model and a flexible model

### 7.1 Import decision tree and metrics

~~~python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Evaluation tools ready")
~~~

### 7.2 Train a shallow decision tree

~~~python
shallow_tree = DecisionTreeRegressor(
    max_depth=2,
    random_state=42
)

shallow_tree.fit(X_train, y_train)

print("Shallow tree trained")
~~~

### 7.3 Evaluate shallow tree on training data

~~~python
train_predictions_shallow = shallow_tree.predict(X_train)

train_mse_shallow = mean_squared_error(
    y_train,
    train_predictions_shallow
)

print("Training MSE:", train_mse_shallow)
~~~

### 7.4 Evaluate shallow tree on validation data

~~~python
valid_predictions_shallow = shallow_tree.predict(X_valid)

valid_mse_shallow = mean_squared_error(
    y_valid,
    valid_predictions_shallow
)

print("Validation MSE:", valid_mse_shallow)
~~~

**Explanation**
A shallow tree may fail to capture important relationships. If both training and validation errors are poor, it may be underfitting.

### 7.5 Train a very deep tree

~~~python
deep_tree = DecisionTreeRegressor(
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

deep_tree.fit(X_train, y_train)

print("Deep tree trained")
~~~

### 7.6 Evaluate deep tree on training data

~~~python
train_predictions_deep = deep_tree.predict(X_train)

train_mse_deep = mean_squared_error(
    y_train,
    train_predictions_deep
)

print("Training MSE:", train_mse_deep)
~~~

### 7.7 Evaluate deep tree on validation data

~~~python
valid_predictions_deep = deep_tree.predict(X_valid)

valid_mse_deep = mean_squared_error(
    y_valid,
    valid_predictions_deep
)

print("Validation MSE:", valid_mse_deep)
~~~

**Explanation**
A very deep tree can fit training observations extremely closely. If validation error is much worse, it may be overfitting.

## 8. Compare train and validation results

### 8.1 Create a comparison table

~~~python
comparison = pd.DataFrame([
    {
        "model": "Shallow tree",
        "train_mse": train_mse_shallow,
        "validation_mse": valid_mse_shallow
    },
    {
        "model": "Deep tree",
        "train_mse": train_mse_deep,
        "validation_mse": valid_mse_deep
    }
])

print(comparison)
~~~

**Expected output**
A two-row table with training and validation error values.

**Explanation**
- Poor training and validation performance can indicate underfitting.
- Very low training error with much higher validation error can indicate overfitting.
- Use the actual results and context; do not diagnose from one number alone.

## 9. Control tree complexity

### 9.1 Train a controlled-depth tree

~~~python
controlled_tree = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

controlled_tree.fit(X_train, y_train)

print("Controlled tree trained")
~~~

### 9.2 Evaluate controlled tree

~~~python
controlled_train = controlled_tree.predict(X_train)
controlled_valid = controlled_tree.predict(X_valid)

print(
    "Train MSE:",
    mean_squared_error(y_train, controlled_train)
)

print(
    "Validation MSE:",
    mean_squared_error(y_valid, controlled_valid)
)
~~~

**Explanation**
Limiting depth and requiring more observations in each leaf can reduce model complexity. The best settings should be selected using cross-validation, not guessed from the test set.

## 10. Learning curves

Learning curves show how model performance changes as the amount of training data increases.

They can help identify:
- Whether more data may improve generalization.
- Whether training and validation performance remain far apart.
- Whether a model is underfitting.
- Whether performance has reached a plateau.

### 10.1 Import learning_curve

~~~python
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor

print("Learning curve function ready")
~~~

### 10.2 Calculate learning curve scores

~~~python
train_sizes, train_scores, valid_scores = learning_curve(
    DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),
    X,
    y,
    cv=5,
    scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.2, 1.0, 5),
    random_state=42
)

print("Training sizes:", train_sizes)
print("Training score shape:", train_scores.shape)
print("Validation score shape:", valid_scores.shape)
~~~

**Explanation**
The function repeats training and validation at different training-set sizes. Scores are negative because scikit-learn uses a larger-is-better scoring convention.

### 10.3 Convert negative errors into positive MSE

~~~python
train_mse_curve = -train_scores.mean(axis=1)
valid_mse_curve = -valid_scores.mean(axis=1)

print("Train MSE:", train_mse_curve)
print("Validation MSE:", valid_mse_curve)
~~~

### 10.4 Plot the learning curves

~~~python
plt.plot(train_sizes, train_mse_curve, marker="o", label="Training MSE")
plt.plot(train_sizes, valid_mse_curve, marker="o", label="Validation MSE")
plt.xlabel("Number of training observations")
plt.ylabel("MSE")
plt.title("Learning Curve")
plt.legend()
plt.show()
~~~

**Explanation**
Use the curves to discuss whether additional training data may help and whether the gap between training and validation performance is closing.

## 11. Practical methods to reduce overfitting

### Regularization
- Penalizes overly complex model behaviour.
- Examples: Ridge and Lasso for linear models.
- Stronger regularization can reduce variance but may increase bias.

### Reduce model complexity
- Limit decision tree depth.
- Increase minimum samples per leaf.
- Reduce unnecessary model flexibility.

### Cross-validation
- Compare model performance across multiple folds.
- Detect unstable performance and support model selection.

### More representative data
- More diverse examples can help the model learn general patterns.
- More data does not fix poor labels or leakage automatically.

### Feature selection
- Remove irrelevant or noisy inputs when evidence supports doing so.
- Fit selection within training folds.

### Early stopping
- Stop iterative training when validation performance stops improving.
- Commonly used with boosting and neural network workflows.
- Monitor validation data without using the final test set for stopping decisions.

## 12. Bias-variance trade-off in model selection

A very simple model may fail to capture meaningful patterns. A very complex model may fit noise.

The objective is not to make the model as simple or as complex as possible. It is to choose a model that performs reliably on unseen data and meets business requirements.

Compare:
- Training performance.
- Validation performance.
- Fold-to-fold variation.
- Model complexity.
- Performance on important subgroups.
- Maintenance and inference cost.

## 13. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Judge model quality from training score | Compare training and validation results |
| Assume a complex model is always better | Validate generalization |
| Reduce complexity without checking performance | Compare candidate settings using cross-validation |
| Add more data without checking quality | Review label accuracy and representativeness |
| Tune repeatedly on the test set | Keep the final test set untouched |
| Ignore fold variation | Report stability across folds |
| Assume overfitting only happens in trees | It can happen in any sufficiently flexible learning process |
| Confuse bias with social bias | Statistical bias and fairness-related bias are different concepts |

## 14. Interview-ready explanations

**What is underfitting?**
- The model is too simple or insufficiently trained to capture important patterns.
- Training and validation performance are both poor.

**What is overfitting?**
- The model learns training-specific details or noise that do not generalize.
- Training performance is strong while validation performance is substantially weaker.

**What is bias?**
- Error associated with overly simple assumptions that miss important patterns.

**What is variance?**
- Sensitivity of a learned model to changes in training data.

**How do you reduce overfitting?**
- Simplify the model, use regularization, validate with cross-validation, improve data quality, use feature selection carefully and apply early stopping where appropriate.

## 15. Scenario-based interview practice

### Scenario 1: Training score is excellent but test score is poor
**Expected response:**
- Suspect overfitting or a train-test distribution mismatch.
- Inspect the split, leakage, model complexity and feature quality.
- Compare cross-validation results.
- Apply regularization or simplify the model and re-evaluate.

### Scenario 2: Both training and validation scores are poor
**Expected response:**
- Suspect underfitting, weak features or data quality issues.
- Check whether the model is too simple.
- Review feature relevance and target quality.
- Compare with a more suitable model.

### Scenario 3: Training and validation performance vary greatly across folds
**Expected response:**
- Check dataset size and subgroup distribution.
- Inspect whether the split strategy reflects time or group structure.
- Review outliers and rare cases.
- Report variation rather than only the mean score.

### Scenario 4: A deep tree performs perfectly on training data
**Expected response:**
- Perfect training fit is not evidence of production quality.
- Compare validation performance.
- Inspect tree complexity and leaf sizes.
- Tune with cross-validation and evaluate on untouched test data.

## 16. Lesson completion checklist

- [ ] I can explain generalization.
- [ ] I can distinguish underfitting and overfitting.
- [ ] I can explain bias and variance.
- [ ] I can compare training and validation performance.
- [ ] I can use learning curves.
- [ ] I can explain regularization and model complexity.
- [ ] I can identify data leakage and validation problems.
- [ ] I can answer overfitting scenarios in interviews.

**Next lesson:** Day 53 — Model Interpretability and Responsible ML.
