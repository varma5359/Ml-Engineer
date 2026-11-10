# Day 48: Cross-Validation and Hyperparameter Tuning

**Pathway:** Machine Learning Engineer  
**Phase:** 5 — Model Evaluation and Feature Engineering  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives
- Explain why one train-validation split may give an unstable estimate.
- Understand K-Fold and Stratified K-Fold cross-validation.
- Understand when time-series splitting is required.
- Use cross_val_score and cross_validate.
- Explain parameters versus hyperparameters.
- Tune a model using GridSearchCV and RandomizedSearchCV.
- Avoid data leakage during preprocessing and tuning.
- Keep the final test set separate from model selection.

## 2. Why cross-validation is needed

A model's evaluation can change depending on which observations are placed in the training and validation sets.

If a dataset is small, one split may not represent the full population. Cross-validation repeats the train-validation process using different folds.

Benefits:
- Uses different portions of the available training data for validation.
- Provides a more stable view than relying on one split.
- Helps compare model settings.
- Reveals whether model performance changes substantially across folds.

Cross-validation does not remove the need for a final untouched test set.

## 3. Understand train, validation and test data

| Dataset | Main purpose |
|---|---|
| Training | Fit model parameters |
| Validation | Compare models and tune hyperparameters |
| Test | Final evaluation after decisions are complete |

A common workflow:
1. Keep the test set separate.
2. Use training data for cross-validation and tuning.
3. Select model and hyperparameters using validation evidence.
4. Refit the selected workflow on the allowed development data.
5. Evaluate once on the untouched test set.

## 4. What is K-Fold cross-validation?

K-Fold divides the available dataset into K groups called folds.

For each iteration:
- One fold is used for validation.
- The remaining folds are used for training.
- The process repeats until every fold has been used for validation once.

For example, with five folds:
- Iteration 1 validates on fold 1.
- Iteration 2 validates on fold 2.
- Continue until fold 5.

The fold scores are summarized to understand model performance and variation.

## 5. Create a sample regression dataset

### 5.1 Import libraries

~~~python
import numpy as np
import pandas as pd

print("Libraries ready")
~~~

### 5.2 Create sample data

~~~python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=300,
    n_features=5,
    noise=15,
    random_state=42
)

print("Features:", X.shape)
print("Target:", y.shape)
~~~

**Expected output**
~~~text
Features: (300, 5)
Target: (300,)
~~~

**Explanation**
We create 300 observations and five input features for a regression example.

### 5.3 Inspect a preview

~~~python
data = pd.DataFrame(
    X,
    columns=[f"feature_{i}" for i in range(1, 6)]
)

data["target"] = y

print(data.head())
~~~

**Expected output**
A five-row DataFrame preview containing five features and the target.

## 6. K-Fold cross-validation

### 6.1 Import KFold

~~~python
from sklearn.model_selection import KFold

print("KFold imported")
~~~

### 6.2 Create five folds

~~~python
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print(kfold)
~~~

**Explanation**
- n_splits defines the number of folds.
- shuffle randomizes observation order before creating folds.
- random_state makes the shuffled fold assignment reproducible.
- For ordered or time-dependent observations, random shuffling may be inappropriate.

### 6.3 Inspect fold sizes

~~~python
for fold_number, (train_index, validation_index) in enumerate(
    kfold.split(X),
    start=1
):
    print(
        "Fold:", fold_number,
        "Train:", len(train_index),
        "Validation:", len(validation_index)
    )
~~~

**Expected output**
Five fold summaries with approximately 240 training rows and 60 validation rows each.

**Explanation**
Each observation is used for validation once and for training in the other folds.

## 7. Evaluate a regression model with cross-validation

### 7.1 Import model and scoring function

~~~python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model = LinearRegression()

print("Model ready")
~~~

### 7.2 Run cross-validation

~~~python
scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="neg_mean_absolute_error"
)

print(scores)
~~~

**Expected output**
Five negative MAE scores.

**Explanation**
Scikit-learn uses a convention where larger scores are better. Error metrics are therefore returned as negative values when used as scoring objectives.

### 7.3 Convert scores into positive MAE values

~~~python
mae_scores = -scores

print("Fold MAE:", mae_scores)
print("Average MAE:", mae_scores.mean())
print("MAE variation:", mae_scores.std())
~~~

**Expected output**
Five positive MAE values, their average and standard deviation.

**Explanation**
- Mean gives a summary of fold performance.
- Standard deviation describes variation between folds.
- A large variation can indicate unstable performance or different difficulty across data subsets.

## 8. Stratified K-Fold for classification

For classification, ordinary K-Fold can produce folds with very different class proportions, especially when the dataset is imbalanced.

**StratifiedKFold** attempts to preserve the class distribution in each fold.

### 8.1 Create a classification dataset

~~~python
from sklearn.datasets import make_classification

X_class, y_class = make_classification(
    n_samples=500,
    n_features=8,
    weights=[0.90, 0.10],
    random_state=42
)

print(pd.Series(y_class).value_counts())
~~~

### 8.2 Import and create StratifiedKFold

~~~python
from sklearn.model_selection import StratifiedKFold

stratified_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Stratified folds ready")
~~~

### 8.3 Inspect class proportions in each fold

~~~python
for fold, (_, validation_index) in enumerate(
    stratified_cv.split(X_class, y_class),
    start=1
):
    fold_labels = y_class[validation_index]

    print(
        "Fold:", fold,
        "Positive:", int((fold_labels == 1).sum()),
        "Total:", len(fold_labels)
    )
~~~

**Explanation**
Each fold is designed to preserve the overall class distribution as closely as possible. It does not guarantee that every fold is representative in every other way.

## 9. Cross-validation for time-series data

Random K-Fold is usually unsuitable when observations are time-dependent.

Why?
- It may train on future observations and validate on past observations.
- This creates unrealistic evaluation.
- Time order must be preserved.

Use time-aware methods such as TimeSeriesSplit when the task is forecasting.

### 9.1 Create an ordered sample

~~~python
time_data = pd.DataFrame({
    "day": range(1, 21),
    "sales": np.arange(20) * 10 + 100
})

print(time_data.head())
~~~

### 9.2 Create TimeSeriesSplit

~~~python
from sklearn.model_selection import TimeSeriesSplit

time_cv = TimeSeriesSplit(n_splits=4)

print("Time split ready")
~~~

### 9.3 Inspect time-based folds

~~~python
for fold, (train_index, validation_index) in enumerate(
    time_cv.split(time_data),
    start=1
):
    print(
        "Fold:", fold,
        "Train days:", time_data.iloc[train_index]["day"].min(),
        "to", time_data.iloc[train_index]["day"].max(),
        "Validation days:", time_data.iloc[validation_index]["day"].min(),
        "to", time_data.iloc[validation_index]["day"].max()
    )
~~~

**Explanation**
Training observations occur before validation observations. This better reflects forecasting future periods from historical data.

## 10. Parameters versus hyperparameters

- **Model parameters** are learned from data during model fitting. Examples include coefficients in linear regression and split rules in a decision tree.
- **Hyperparameters** are settings chosen before or during model selection. Examples include tree depth, number of neighbours and regularization strength.

Hyperparameter tuning searches for settings that perform well under a chosen validation strategy.

## 11. GridSearchCV

GridSearchCV evaluates every specified combination of hyperparameter values using cross-validation.

Use it when:
- The search space is small.
- You have a focused set of candidate values.
- You can afford to evaluate all combinations.

### 11.1 Import a decision tree and GridSearchCV

~~~python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

print("Grid search tools ready")
~~~

### 11.2 Define a model

~~~python
tree_model = DecisionTreeRegressor(random_state=42)

print(tree_model)
~~~

### 11.3 Define candidate hyperparameters

~~~python
parameter_grid = {
    "max_depth": [2, 3, 5, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

print(parameter_grid)
~~~

**Explanation**
- max_depth limits tree depth.
- min_samples_split controls the minimum observations required to split a node.
- min_samples_leaf controls the minimum observations required in a leaf.
- Candidate values should be reasonable for the dataset and model.

### 11.4 Create GridSearchCV

~~~python
grid_search = GridSearchCV(
    estimator=tree_model,
    param_grid=parameter_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("Grid search configured")
~~~

**Explanation**
- estimator is the model being tuned.
- param_grid contains candidate settings.
- cv defines the cross-validation strategy or fold count.
- scoring defines the selection metric.
- n_jobs controls parallel execution; -1 uses available CPU cores.

### 11.5 Fit the search

~~~python
grid_search.fit(X, y)

print("Grid search complete")
~~~

### 11.6 Inspect the best settings

~~~python
print("Best parameters:", grid_search.best_params_)
print("Best CV MAE:", -grid_search.best_score_)
~~~

**Explanation**
The best score is negative because the search used negative MAE. Convert it to a positive error value for reporting.

### 11.7 Inspect the best estimator

~~~python
best_tree = grid_search.best_estimator_

print(best_tree)
~~~

**Explanation**
best_estimator_ is the model refitted using the selected hyperparameters according to GridSearchCV's settings.

## 12. RandomizedSearchCV

RandomizedSearchCV samples a fixed number of parameter combinations rather than evaluating every possible combination.

Use it when:
- The search space is large.
- Evaluating every combination is expensive.
- You want to explore a broader range with a limited compute budget.

### 12.1 Import RandomizedSearchCV

~~~python
from sklearn.model_selection import RandomizedSearchCV

print("Randomized search imported")
~~~

### 12.2 Define a parameter distribution

~~~python
parameter_distributions = {
    "max_depth": [2, 3, 4, 5, 8, 10, None],
    "min_samples_split": [2, 4, 6, 8, 10, 15],
    "min_samples_leaf": [1, 2, 3, 4, 6]
}

print(parameter_distributions)
~~~

### 12.3 Configure randomized search

~~~python
random_search = RandomizedSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_distributions=parameter_distributions,
    n_iter=15,
    cv=5,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1
)

print("Random search configured")
~~~

### 12.4 Fit randomized search

~~~python
random_search.fit(X, y)

print("Random search complete")
~~~

### 12.5 Inspect selected settings

~~~python
print("Best parameters:", random_search.best_params_)
print("Best CV MAE:", -random_search.best_score_)
~~~

**Explanation**
The search evaluates a limited sample of combinations. It may be more efficient than a full grid, but it does not guarantee the global best setting.

## 13. Prevent leakage during cross-validation

Preprocessing must be fitted separately inside each training fold.

Incorrect approach:
- Scale the entire dataset first.
- Run cross-validation on the already transformed dataset.
- Validation-fold information has influenced the preprocessing transformation.

Better approach:
- Put imputation, scaling, feature selection and the estimator inside a Pipeline.
- Cross-validation fits the pipeline independently on each training fold.
- Validation folds receive transformations learned only from their corresponding training folds.

### 13.1 Create a pipeline

~~~python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ridge_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

print(ridge_pipeline)
~~~

### 13.2 Cross-validate the pipeline

~~~python
pipeline_scores = cross_val_score(
    ridge_pipeline,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

print("Fold MAE:", -pipeline_scores)
print("Average MAE:", -pipeline_scores.mean())
~~~

**Explanation**
The pipeline ensures that preprocessing is learned separately within each fold. This is safer than fitting transformations once on all observations.

## 14. Choose a tuning strategy

| Situation | Recommended approach |
|---|---|
| Small focused parameter space | GridSearchCV |
| Large parameter space | RandomizedSearchCV |
| Time-dependent forecasting | TimeSeriesSplit or suitable time-aware validation |
| Imbalanced classification | StratifiedKFold |
| Preprocessing required | Pipeline inside cross-validation |
| Very limited data | Carefully designed cross-validation and clear uncertainty reporting |

## 15. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Tune using the test set | Tune on training/validation data and preserve final test |
| Use random folds for time series | Preserve time order |
| Scale data before cross-validation | Put scaling inside a Pipeline |
| Ignore class imbalance in folds | Use stratified splitting when appropriate |
| Search too many combinations without a plan | Define a focused search space |
| Choose the best single fold | Compare mean and fold variation |
| Forget negative scoring convention | Convert negative error scores to positive values for reporting |
| Treat best parameters as universally best | Validate on representative unseen data |

## 16. Interview-ready explanations

**What is cross-validation?**
- A method that repeats training and validation using different folds.
- It gives a more reliable view than depending on one random split.
- It helps compare models and hyperparameters.

**What is K-Fold?**
- Divide data into K folds.
- Train on K-1 folds and validate on the remaining fold.
- Repeat until every fold has been used for validation.

**What is Stratified K-Fold?**
- A classification splitting method that tries to preserve class proportions in each fold.
- It is useful when classes are imbalanced.

**What is hyperparameter tuning?**
- Searching for model settings that produce good validation performance.
- Examples include tree depth, number of neighbours and regularization strength.

**GridSearchCV versus RandomizedSearchCV?**
- GridSearchCV evaluates all specified combinations.
- RandomizedSearchCV samples a fixed number of combinations.
- Choose based on search-space size and compute budget.

## 17. Scenario-based interview practice

### Scenario 1: One train-test split gives a very different score from another
**Expected response:**
- Performance may depend on which observations enter each split.
- Use cross-validation to examine performance across folds.
- Check whether the data contains groups, time order or class imbalance that requires a special split.

### Scenario 2: Cross-validation scores vary widely
**Expected response:**
- Inspect fold-level results.
- Check sample size, outliers, class distribution and subgroup differences.
- Verify that splitting matches the data-generating process.
- Report variability rather than only the average.

### Scenario 3: A developer scales all rows before cross-validation
**Expected response:**
- This can leak information from validation folds into preprocessing.
- Put the scaler inside a Pipeline.
- Let cross-validation fit preprocessing on each training fold only.

### Scenario 4: The team uses GridSearchCV with thousands of combinations
**Expected response:**
- The search may be computationally expensive.
- Reduce the search space or use RandomizedSearchCV.
- Prioritize plausible settings and track compute budget.
- Validate the selected configuration on untouched data.

### Scenario 5: A forecasting model uses shuffled K-Fold
**Expected response:**
- Random folds can train on future observations and validate on earlier observations.
- Use time-aware validation such as TimeSeriesSplit.
- Ensure feature generation also respects the prediction timestamp.

## 18. Lesson completion checklist

- [ ] I can explain K-Fold cross-validation.
- [ ] I can distinguish K-Fold and Stratified K-Fold.
- [ ] I understand time-series splitting.
- [ ] I can use cross_val_score.
- [ ] I can explain model parameters and hyperparameters.
- [ ] I can configure GridSearchCV and RandomizedSearchCV.
- [ ] I can prevent preprocessing leakage with a Pipeline.
- [ ] I can interpret fold variation and best parameters.

**Next lesson:** Day 49 — Feature Selection and Feature Importance.
