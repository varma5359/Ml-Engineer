# Day 30: Random Forest and Ensemble Learning

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain ensemble learning in simple language.
- Understand why combining multiple models can improve stability.
- Explain how Random Forest combines decision trees.
- Understand bootstrap sampling and random feature selection.
- Train RandomForestClassifier and RandomForestRegressor.
- Tune n_estimators, max_depth and max_features.
- Inspect feature importance cautiously.
- Understand out-of-bag evaluation conceptually.
- Compare a single tree with a Random Forest.
- Recognize Random Forest strengths and limitations.

## 2. What is ensemble learning?

**Ensemble learning** combines predictions from multiple models to produce one final prediction.

The idea is similar to asking several independent reviewers instead of relying on one opinion. If the models make different errors, combining them can create a more stable result.

Common ensemble approaches:
- Bagging: train models on different samples and combine their predictions.
- Boosting: train models sequentially so later models focus on errors made earlier.
- Voting: combine predictions from different model types.
- Stacking: use another model to learn how to combine base-model predictions.

Random Forest is a bagging-based ensemble of decision trees.

## 3. What is Random Forest?

**Random Forest** trains many decision trees using different bootstrap samples and randomized feature selection. It combines the trees' predictions.

For classification, trees vote for a class. For regression, tree predictions are combined, commonly by averaging.

Random Forest can reduce the instability of a single decision tree and often provides a strong baseline for tabular data.

## 4. Understand bootstrap sampling

Bootstrap sampling creates a training sample by selecting observations randomly with replacement.

This means:
- Some original rows may appear more than once in a bootstrap sample.
- Some original rows may not appear in that particular sample.
- Different trees can be trained on different samples.

The trees therefore see different versions of the training data.

## 5. Understand random feature selection

At each split, a tree can consider a randomly selected subset of available features rather than always considering every feature.

This encourages diversity among trees. If every tree were identical, combining them would provide little benefit.

## 6. Create a classification dataset

### 6.1 Import libraries

### 6.1. Import Pandas, NumPy and plotting libraries

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

These libraries prepare and inspect the classification data.



### 6.2 Generate a classification dataset

### 6.2. Create synthetic customer outcomes

```python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=500,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    weights=[0.65, 0.35],
    random_state=42
)

print(X_values.shape)
print(pd.Series(y).value_counts())
```

**Expected output**

```text
(500, 8) and class counts
```

**Explanation**

The data contains eight features and a moderately imbalanced binary target. It is synthetic teaching data.



### 6.3 Create a DataFrame

### 6.3. Assign feature names

```python
feature_names = [f"feature_{i}" for i in range(1, 9)]

df = pd.DataFrame(X_values, columns=feature_names)
df["target"] = y

print(df.head())
```

**Expected output**

```text
A five-row preview with eight features and target
```

**Explanation**

Feature names make the model inputs easier to inspect.



### 6.4 Inspect the target distribution

### 6.4. Count target classes

```python
print(df["target"].value_counts(normalize=True))
```

**Expected output**

```text
The proportion of each target class
```

**Explanation**

The class distribution helps determine which metrics and split strategy to use.



## 7. Split the dataset

### 7.1 Create X and y

### 7.1. Separate features and target

```python
X = df.drop(columns=["target"])
y = df["target"]

print(X.shape, y.shape)
```

**Expected output**

```text
(500, 8) and (500,)
```

**Explanation**

The target is not included among the input features.



### 7.2 Create a stratified split

### 7.2. Split training and test data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
(400, 8) and (100, 8)
```

**Explanation**

The test set is held back. Stratification preserves approximate class proportions.



## 8. Train a single Decision Tree baseline

### 8.1 Import DecisionTreeClassifier

### 8.1. Import the decision tree

```python
from sklearn.tree import DecisionTreeClassifier

print("Decision tree imported")
```

**Expected output**

```text
Decision tree imported
```

**Explanation**

A single tree provides a useful comparison against the ensemble.



### 8.2 Fit the decision tree

### 8.2. Train a decision tree

```python
tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

print("Tree fitted")
```

**Expected output**

```text
Tree fitted
```

**Explanation**

The model learns one decision tree from the training data.



### 8.3 Predict test labels

### 8.3. Generate tree predictions

```python
tree_predictions = tree_model.predict(X_test)

print(tree_predictions[:10])
```

**Expected output**

```text
The first ten predicted class labels
```

**Explanation**

The tree follows learned feature splits to assign each class.



### 8.4 Evaluate tree accuracy

### 8.4. Calculate tree accuracy

```python
from sklearn.metrics import accuracy_score

tree_accuracy = accuracy_score(y_test, tree_predictions)

print(tree_accuracy)
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

This is the single-tree reference score. Also inspect class-level metrics.



## 9. Train a Random Forest classifier

### 9.1 Import RandomForestClassifier

### 9.1. Import RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier

print("Random Forest classifier imported")
```

**Expected output**

```text
Random Forest classifier imported
```

**Explanation**

The estimator combines predictions from multiple decision trees.



### 9.2 Initialize a Random Forest

### 9.2. Create a forest with 100 trees

```python
forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print(forest)
```

**Expected output**

```text
RandomForestClassifier(n_estimators=100, random_state=42)
```

**Explanation**

n_estimators controls how many trees are built. A larger forest can improve stability but uses more compute and memory.



### 9.3 Fit the forest

### 9.3. Train Random Forest

```python
forest.fit(X_train, y_train)

print("Random Forest fitted")
```

**Expected output**

```text
Random Forest fitted
```

**Explanation**

Each tree is trained using the Random Forest sampling and feature-selection process.



### 9.4 Predict test labels

### 9.4. Generate forest predictions

```python
forest_predictions = forest.predict(X_test)

print(forest_predictions[:10])
```

**Expected output**

```text
The first ten predicted class labels
```

**Explanation**

The final class prediction combines the votes from the trees.



### 9.5 Predict class probabilities

### 9.5. Generate class probabilities

```python
forest_probabilities = forest.predict_proba(X_test)

print(forest_probabilities[:5])
```

**Expected output**

```text
A probability matrix for the first five observations
```

**Explanation**

The probabilities are derived from the ensemble's tree predictions. Inspect classes_ to confirm column order.



### 9.6 Evaluate Random Forest accuracy

### 9.6. Calculate forest accuracy

```python
forest_accuracy = accuracy_score(y_test, forest_predictions)

print(forest_accuracy)
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

Compare this with the single decision tree on the same held-out observations.



### 9.7 Compare tree and forest

### 9.7. Create a model comparison

```python
comparison = pd.DataFrame({
    "model": ["Decision Tree", "Random Forest"],
    "accuracy": [tree_accuracy, forest_accuracy]
})

print(comparison)
```

**Expected output**

```text
A two-row accuracy comparison
```

**Explanation**

This is one test split. Use cross-validation and class-specific metrics for stronger conclusions.



## 10. Inspect Random Forest parameters

### 10.1 Inspect number of trees

### 10.1. Read n_estimators

```python
print(len(forest.estimators_))
```

**Expected output**

```text
100
```

**Explanation**

The fitted forest contains 100 decision trees.



### 10.2 Inspect one tree's depth

### 10.2. Read the depth of the first tree

```python
print(forest.estimators_[0].get_depth())
```

**Expected output**

```text
An integer depth value
```

**Explanation**

Each tree may have a different structure and depth.



### 10.3 Inspect the number of leaves

### 10.3. Read leaves in the first tree

```python
print(forest.estimators_[0].get_n_leaves())
```

**Expected output**

```text
An integer leaf count
```

**Explanation**

The number of leaves gives a simple view of tree complexity.



## 11. Understand important Random Forest parameters

| Parameter | Meaning | Effect to consider |
|---|---|---|
| n_estimators | Number of trees | More trees increase computation and can stabilize predictions |
| max_depth | Maximum tree depth | Limits tree complexity |
| min_samples_split | Minimum observations needed to split | Larger values restrict splitting |
| min_samples_leaf | Minimum observations in a leaf | Larger values create smoother leaves |
| max_features | Number or fraction of features considered at each split | Controls tree diversity |
| bootstrap | Whether bootstrap samples are used | Default is generally True |
| class_weight | Class weighting option | Can help when class importance is uneven |

### 11.1 Train a shallow forest

### 11.1. Limit tree depth

```python
shallow_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

shallow_forest.fit(X_train, y_train)

print("Shallow forest fitted")
```

**Expected output**

```text
Shallow forest fitted
```

**Explanation**

max_depth limits the depth of each tree. This can reduce complexity but may also underfit.



### 11.2 Train a forest with larger leaf size

### 11.2. Set min_samples_leaf

```python
leaf_forest = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=5,
    random_state=42
)

leaf_forest.fit(X_train, y_train)

print("Leaf-constrained forest fitted")
```

**Expected output**

```text
Leaf-constrained forest fitted
```

**Explanation**

Each terminal leaf must contain at least the configured number of training observations.



### 11.3 Compare max_features settings

### 11.3. Train forests with different feature sampling

```python
feature_sampling_results = []

for setting in ["sqrt", "log2", None]:
    candidate = RandomForestClassifier(
        n_estimators=100,
        max_features=setting,
        random_state=42
    )

    candidate.fit(X_train, y_train)

    feature_sampling_results.append({
        "max_features": str(setting),
        "test_accuracy": candidate.score(X_test, y_test)
    })

print(pd.DataFrame(feature_sampling_results))
```

**Expected output**

```text
A table comparing test accuracy for different feature-sampling settings
```

**Explanation**

In real model selection, tune these settings using validation or cross-validation and reserve test data for final evaluation.



## 12. Feature importance

Random Forest provides feature importance values based on how features contribute to impurity reduction across trees.

These values are useful for exploration but have limitations:
- They are not causal effects.
- They can favour continuous or high-cardinality features.
- Correlated features may share or distort importance.
- They do not prove that a feature will generalize.

### 12.1 Read feature importance

### 12.1. Create a feature importance table

```python
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": forest.feature_importances_
}).sort_values("importance", ascending=False)

print(importance)
```

**Expected output**

```text
Features ordered by impurity-based importance
```

**Explanation**

This summarizes how the fitted forest used features for splits.



### 12.2 Plot feature importance

### 12.2. Visualize feature importance

```python
sns.barplot(
    data=importance,
    x="importance",
    y="feature"
)
plt.title("Random Forest feature importance")
plt.show()
```

**Expected output**

```text
A horizontal bar chart of feature importance
```

**Explanation**

Use the chart as a starting point for investigation, not as proof of business causation.



### 12.3 Compare with permutation importance

### 12.3. Calculate permutation importance

```python
from sklearn.inspection import permutation_importance

permutation_result = permutation_importance(
    forest,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

permutation_table = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": permutation_result.importances_mean,
    "importance_std": permutation_result.importances_std
}).sort_values("importance_mean", ascending=False)

print(permutation_table)
```

**Expected output**

```text
A table showing mean and variation in permutation importance
```

**Explanation**

Permutation importance measures how model performance changes when a feature is shuffled. Calculate it on validation data for model development or on held-out test data for a final explanatory analysis, without using it to tune the model on that same test set.



## 13. Out-of-bag evaluation

When bootstrap sampling is enabled, some training observations are left out of each individual tree's bootstrap sample. These are called out-of-bag observations for that tree.

Random Forest can use these left-out observations to estimate an out-of-bag score.

### 13.1 Train with out-of-bag scoring

### 13.1. Enable out-of-bag score

```python
oob_forest = RandomForestClassifier(
    n_estimators=200,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

oob_forest.fit(X_train, y_train)

print(oob_forest.oob_score_)
```

**Expected output**

```text
An out-of-bag score between zero and one
```

**Explanation**

OOB scoring uses training observations that were not included in each tree's bootstrap sample. It is a useful internal estimate, not a replacement for an independent test set.



### 13.2 Inspect OOB predictions

### 13.2. Read out-of-bag decision values

```python
print(oob_forest.oob_decision_function_[:5])
```

**Expected output**

```text
The first five rows of OOB class probability estimates
```

**Explanation**

These values are aggregated from trees for which each observation was out-of-bag. Some small datasets can have missing OOB predictions if too few trees are used.



## 14. Random Forest for regression

Random Forest can also predict numerical targets by combining regression-tree predictions.

### 14.1 Create a house-price dataset

### 14.1. Prepare area and bedroom features

```python
reg_df = pd.DataFrame({
    "area": [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
             1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300],
    "bedrooms": [1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                 3, 4, 4, 4, 4, 4, 5, 5],
    "price": [20, 22, 25, 28, 31, 34, 37, 40, 43, 46,
              49, 53, 56, 59, 63, 66, 70, 74]
})

print(reg_df.head())
```

**Expected output**

```text
A house-price feature table
```

**Explanation**

The target is price and the inputs are area and bedrooms.



### 14.2 Split regression data

### 14.2. Create regression X and y

```python
X_reg = reg_df[["area", "bedrooms"]]
y_reg = reg_df["price"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

print(X_train_r.shape, X_test_r.shape)
```

**Expected output**

```text
Training and test shapes
```

**Explanation**

The test rows are held out for evaluation.



### 14.3 Train RandomForestRegressor

### 14.3. Import and fit regression forest

```python
from sklearn.ensemble import RandomForestRegressor

forest_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

forest_regressor.fit(X_train_r, y_train_r)

print("Regression forest fitted")
```

**Expected output**

```text
Regression forest fitted
```

**Explanation**

The estimator trains multiple regression trees and combines their numerical predictions.



### 14.4 Predict prices

### 14.4. Generate regression predictions

```python
reg_predictions = forest_regressor.predict(X_test_r)

print(reg_predictions)
```

**Expected output**

```text
Numerical house-price predictions
```

**Explanation**

The forest combines the predictions from its individual trees.



### 14.5 Evaluate regression error

### 14.5. Calculate regression MAE

```python
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test_r, reg_predictions))
```

**Expected output**

```text
A non-negative MAE value
```

**Explanation**

The metric is expressed in the target's price units.



## 15. Strengths and limitations

### Strengths
- Often strong on tabular datasets.
- Captures nonlinear relationships and interactions.
- Less sensitive to feature scaling than distance-based models.
- Combines many trees to reduce instability of one tree.
- Supports classification and regression.
- Provides useful exploratory feature importance.

### Limitations
- Can require more memory and computation than one tree.
- Predictions can be slower with many trees.
- Model explanation is less straightforward than a shallow tree.
- Impurity-based feature importance can be biased.
- Large forests may be difficult to deploy in constrained environments.
- Extrapolation in regression is limited by the target values represented in terminal leaves.

## 16. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Assume more trees always improve business value | Compare performance and inference cost |
| Use unlimited depth without validation | Tune depth and leaf constraints |
| Treat feature importance as causation | Interpret as model-specific diagnostic |
| Ignore class imbalance | Inspect class metrics and consider class_weight |
| Tune on test data | Use validation or cross-validation |
| Save only the estimator when preprocessing exists | Save the full pipeline |
| Ignore inference latency and memory | Measure operational cost |
| Assume Random Forest extrapolates like linear regression | Check prediction range and tree-leaf behaviour |

## 17. Interview-ready explanations

**What is ensemble learning?**  
It combines predictions from multiple models to produce a final prediction that can be more stable or accurate than an individual model.

**What is Random Forest?**  
It is an ensemble of decision trees trained with bootstrap samples and randomized feature selection. Classification uses voting; regression combines numerical predictions.

**What is bootstrap sampling?**  
It creates training samples by randomly selecting observations with replacement, so different trees see different samples.

**Why does Random Forest use random feature selection?**  
It encourages diversity among trees so the ensemble is not simply repeating the same decision tree.

**What is out-of-bag evaluation?**  
It uses training observations left out of a tree's bootstrap sample to estimate performance. It is useful but does not replace an independent test set.

## 18. Scenario questions

1. A single decision tree has unstable performance across splits. What could you try?  
   **Answer:** Random Forest can reduce the instability by combining trees trained on different samples and feature subsets.

2. Increasing n_estimators makes training and prediction slower. What do you consider?  
   **Answer:** Compare the performance gain with memory, training and inference cost. More trees are not automatically worth the added cost.

3. Random Forest feature importance ranks customer ID highly. What do you investigate?  
   **Answer:** Check whether the ID encodes ordering, source or leakage, and whether the feature generalizes. Do not assume it is meaningful.

4. A forest has excellent training accuracy but poor test accuracy. What do you inspect?  
   **Answer:** Tree depth, leaf size, data leakage, split strategy and distribution differences.

5. A Random Forest regressor predicts a house price beyond the maximum training target. Is that expected?  
   **Answer:** Standard tree regression predictions are based on target values in leaves, so predictions generally do not extrapolate beyond the training target range in the same way as linear regression.

## 19. Lesson completion checklist

- [ ] I can explain ensemble learning.
- [ ] I understand bootstrap sampling and random feature selection.
- [ ] I can train RandomForestClassifier.
- [ ] I can compare it with a single decision tree.
- [ ] I understand n_estimators and tree-complexity settings.
- [ ] I can inspect feature importance carefully.
- [ ] I understand out-of-bag evaluation.
- [ ] I can train RandomForestRegressor.
- [ ] I can explain strengths and limitations.

**Next lesson:** Day 31 — Support Vector Machines (SVM).
