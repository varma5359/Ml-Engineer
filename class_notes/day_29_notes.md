# Day 29: Decision Trees and Splitting Criteria

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised Learning  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain decision trees in simple language.
- Understand root, decision node, branch and leaf.
- Explain how a tree chooses splits conceptually.
- Distinguish classification and regression trees.
- Train DecisionTreeClassifier using scikit-learn.
- Understand Gini impurity and entropy at a conceptual level.
- Control tree complexity using depth and leaf settings.
- Visualize and interpret a small decision tree.
- Recognize overfitting in deep trees.
- Compare tree performance with a baseline.

## 2. What is a Decision Tree?

A **Decision Tree** is a supervised learning algorithm that makes predictions through a sequence of questions about input features.

Example:
- Is customer tenure less than six months?
- Are support calls greater than three?
- Is the contract monthly?

The tree follows the branch that matches the observation and eventually reaches a leaf node containing a prediction.

Decision trees are often easy to explain because the prediction can be described as a set of rules.

## 3. Important tree vocabulary

| Term | Meaning |
|---|---|
| Root node | First decision point |
| Decision node | A feature-based question |
| Branch | One outcome of a decision |
| Leaf node | Final prediction |
| Depth | Number of decision levels along a path |
| Split | Dividing observations into child groups |
| Impurity | How mixed the target classes are in a node |
| Pruning | Reducing tree complexity |

## 4. Create a classification dataset

### 4.1 Import libraries

### 4.1. Import Pandas, NumPy and plotting tools

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

These libraries support data preparation and visualization.



### 4.2 Generate a two-class dataset

### 4.2. Create customer classification data

```python
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=300,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

print(X_values.shape)
```

**Expected output**

```text
(300, 4)
```

**Explanation**

The dataset contains 300 observations and four numerical features.



### 4.3 Create named columns

### 4.3. Build the DataFrame

```python
df = pd.DataFrame(
    X_values,
    columns=["tenure", "spend", "support_calls", "usage"]
)

df["target"] = y

print(df.head())
```

**Expected output**

```text
A five-row preview with four features and one target
```

**Explanation**

The target is kept separate from the feature columns during training.



### 4.4 Inspect target counts

### 4.4. Count target classes

```python
print(df["target"].value_counts())
```

**Expected output**

```text
Counts for class 0 and class 1
```

**Explanation**

Understand class balance before selecting a split and evaluation metric.



## 5. Split the data

### 5.1 Separate features and target

### 5.1. Create X and y

```python
X = df.drop(columns=["target"])
y = df["target"]

print(X.shape, y.shape)
```

**Expected output**

```text
(300, 4) and (300,)
```

**Explanation**

X contains four features and y contains the target label.



### 5.2 Create a stratified train-test split

### 5.2. Split the data

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
(240, 4) and (60, 4)
```

**Explanation**

The test set is held back for evaluation. Stratification preserves approximate class proportions.



## 6. Train a basic Decision Tree

### 6.1 Import DecisionTreeClassifier

### 6.1. Import the classifier

```python
from sklearn.tree import DecisionTreeClassifier

print("DecisionTreeClassifier imported")
```

**Expected output**

```text
DecisionTreeClassifier imported
```

**Explanation**

This estimator creates a tree of feature-based decision rules.



### 6.2 Initialize a tree

### 6.2. Create a tree with default settings

```python
tree = DecisionTreeClassifier(random_state=42)

print(tree)
```

**Expected output**

```text
DecisionTreeClassifier(random_state=42)
```

**Explanation**

The model is initialized. A fully grown tree can become complex, so we will inspect and control its depth.



### 6.3 Fit the tree

### 6.3. Train the decision tree

```python
tree.fit(X_train, y_train)

print("Tree fitted")
```

**Expected output**

```text
Tree fitted
```

**Explanation**

The estimator learns splits from the training data.



### 6.4 Predict test labels

### 6.4. Generate predictions

```python
tree_predictions = tree.predict(X_test)

print(tree_predictions[:15])
```

**Expected output**

```text
The first fifteen predicted class labels
```

**Explanation**

Each prediction follows a path through the learned decision rules.



### 6.5 Evaluate accuracy

### 6.5. Calculate test accuracy

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, tree_predictions))
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

Accuracy is one summary. Also inspect class-level metrics and error patterns.



### 6.6 Print a classification report

### 6.6. Review precision and recall

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    tree_predictions,
    zero_division=0
))
```

**Expected output**

```text
A class-wise classification report
```

**Explanation**

This shows how well the tree identifies each class.



## 7. Understand how a tree splits

A tree evaluates possible feature splits and chooses a split that improves class separation according to its selected criterion.

For classification, a node is more pure when most observations belong to one class. A useful split makes child nodes more class-pure than the parent.

### Gini impurity

Gini impurity describes how mixed the class labels are within a node. A pure node containing one class has low impurity.

### Entropy

Entropy measures uncertainty or disorder in the class distribution. A pure node has low entropy.

You do not need to memorize formulas for this lesson. Understand that the criterion helps the algorithm choose useful splits.

### 7.1 Train using Gini

### 7.1. Create a Gini-based tree

```python
gini_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

gini_tree.fit(X_train, y_train)

print("Gini tree fitted")
```

**Expected output**

```text
Gini tree fitted
```

**Explanation**

Gini is the default splitting criterion in many scikit-learn decision-tree classification workflows.



### 7.2 Train using entropy

### 7.2. Create an entropy-based tree

```python
entropy_tree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

entropy_tree.fit(X_train, y_train)

print("Entropy tree fitted")
```

**Expected output**

```text
Entropy tree fitted
```

**Explanation**

Entropy is another criterion for measuring class uncertainty during split selection.



### 7.3 Compare criteria

### 7.3. Evaluate both trees

```python
for name, estimator in [
    ("Gini", gini_tree),
    ("Entropy", entropy_tree)
]:
    prediction = estimator.predict(X_test)
    print(name, accuracy_score(y_test, prediction))
```

**Expected output**

```text
A test accuracy value for each criterion
```

**Explanation**

The best criterion depends on the dataset. Differences may be small and should be evaluated using a consistent split or cross-validation.



## 8. Visualize a decision tree

### 8.1 Import plot_tree

### 8.1. Import the tree plotting function

```python
from sklearn.tree import plot_tree

print("Tree plotting function ready")
```

**Expected output**

```text
Tree plotting function ready
```

**Explanation**

plot_tree creates a visual representation of learned decision rules.



### 8.2 Plot a shallow tree

### 8.2. Create a shallow decision tree

```python
small_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

small_tree.fit(X_train, y_train)

print("Shallow tree fitted")
```

**Expected output**

```text
Shallow tree fitted
```

**Explanation**

A shallow tree is easier to interpret and less likely to memorize every training observation.



### 8.3 Visualize the shallow tree

### 8.3. Plot tree rules

```python
plt.figure(figsize=(16, 8))
plot_tree(
    small_tree,
    feature_names=X.columns,
    class_names=["Class 0", "Class 1"],
    filled=True,
    rounded=True
)
plt.title("Decision tree with maximum depth 3")
plt.show()
```

**Expected output**

```text
A diagram showing decision nodes, branches and leaf predictions
```

**Explanation**

Each node displays a feature condition and information about the observations reaching that node. The chart can become crowded for deep trees.



## 9. Control tree complexity

Important parameters:
- max_depth: maximum number of decision levels.
- min_samples_split: minimum observations needed before a node can be split.
- min_samples_leaf: minimum observations required in a leaf.
- max_leaf_nodes: maximum number of leaf nodes.
- ccp_alpha: pruning strength for cost-complexity pruning.

### 9.1 Limit maximum depth

### 9.1. Train a depth-two tree

```python
depth_two = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

depth_two.fit(X_train, y_train)

print(depth_two.get_depth())
```

**Expected output**

```text
2 or less
```

**Explanation**

The maximum depth limits the number of decision levels. The actual depth may be smaller if no further useful split is available.



### 9.2 Set minimum samples per leaf

### 9.2. Require more observations in each leaf

```python
leaf_tree = DecisionTreeClassifier(
    min_samples_leaf=10,
    random_state=42
)

leaf_tree.fit(X_train, y_train)

print(leaf_tree.get_n_leaves())
```

**Expected output**

```text
An integer leaf count
```

**Explanation**

A larger minimum leaf size prevents the tree from creating very small terminal groups.



### 9.3 Set minimum samples to split

### 9.3. Control split eligibility

```python
split_tree = DecisionTreeClassifier(
    min_samples_split=20,
    random_state=42
)

split_tree.fit(X_train, y_train)

print(split_tree.get_depth())
```

**Expected output**

```text
The learned tree depth
```

**Explanation**

A node must contain enough observations before the algorithm can consider splitting it.



### 9.4 Compare depth and model size

### 9.4. Compare trees with different depths

```python
depth_results = []

for depth in [1, 2, 3, 5, 10, None]:
    candidate = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    candidate.fit(X_train, y_train)

    depth_results.append({
        "max_depth": str(depth),
        "actual_depth": candidate.get_depth(),
        "leaves": candidate.get_n_leaves(),
        "train_accuracy": candidate.score(X_train, y_train),
        "test_accuracy": candidate.score(X_test, y_test)
    })

print(pd.DataFrame(depth_results))
```

**Expected output**

```text
A table comparing depth, leaf count, training accuracy and test accuracy
```

**Explanation**

A very deep tree may fit training data extremely well while failing to improve test performance. Select complexity using validation or cross-validation.



## 10. Feature importance

Decision trees can report impurity-based feature importance. It summarizes how much each feature contributed to reducing impurity across the fitted tree.

It is not a causal explanation and can favour features with many possible split points.

### 10.1 Read feature importance

### 10.1. Inspect feature importance values

```python
importance_table = pd.DataFrame({
    "feature": X.columns,
    "importance": small_tree.feature_importances_
}).sort_values("importance", ascending=False)

print(importance_table)
```

**Expected output**

```text
A table of feature importance values
```

**Explanation**

The values are specific to the fitted tree. They do not prove that a feature causes the target.



### 10.2 Plot feature importance

### 10.2. Visualize feature importance

```python
sns.barplot(
    data=importance_table,
    x="importance",
    y="feature"
)
plt.title("Decision tree feature importance")
plt.show()
```

**Expected output**

```text
A horizontal bar chart of feature importance
```

**Explanation**

This gives a visual summary of the fitted tree's split usage. Validate importance stability and consider permutation importance for another perspective.



## 11. Decision Tree for regression

Decision trees can also predict numerical values. A regression tree splits observations into groups and predicts a value for each leaf, commonly based on the training targets in that leaf.

### 11.1 Create a regression dataset

### 11.1. Prepare area and price data

```python
reg_df = pd.DataFrame({
    "area": [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
             1500, 1600, 1700, 1800, 1900, 2000],
    "price": [20, 22, 25, 27, 30, 33, 35, 38, 41, 44,
              47, 50, 53, 56, 59, 63]
})

print(reg_df.head())
```

**Expected output**

```text
A small numerical regression dataset
```

**Explanation**

The target is house price and the input is area.



### 11.2 Separate X and y

### 11.2. Create regression features and target

```python
X_reg = reg_df[["area"]]
y_reg = reg_df["price"]

print(X_reg.shape, y_reg.shape)
```

**Expected output**

```text
(16, 1) and (16,)
```

**Explanation**

The feature table contains one numerical input and the target contains prices.



### 11.3 Split regression data

### 11.3. Create regression train-test split

```python
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.25,
    random_state=42
)

print(X_train_r.shape, X_test_r.shape)
```

**Expected output**

```text
(12, 1) and (4, 1)
```

**Explanation**

The test data is held back for evaluation.



### 11.4 Train DecisionTreeRegressor

### 11.4. Fit a regression tree

```python
from sklearn.tree import DecisionTreeRegressor

tree_regressor = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

tree_regressor.fit(X_train_r, y_train_r)

print("Regression tree fitted")
```

**Expected output**

```text
Regression tree fitted
```

**Explanation**

The model creates regions of area values and predicts a numerical price for each region.



### 11.5 Predict house prices

### 11.5. Generate regression predictions

```python
reg_predictions = tree_regressor.predict(X_test_r)

print(reg_predictions)
```

**Expected output**

```text
Predicted house prices
```

**Explanation**

The output is a numerical target rather than a class label.



### 11.6 Evaluate regression error

### 11.6. Calculate MAE

```python
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test_r, reg_predictions))
```

**Expected output**

```text
A non-negative MAE value
```

**Explanation**

MAE is expressed in the target's price units.



## 12. Strengths and limitations

### Strengths
- Easy to explain as a sequence of rules.
- Can model nonlinear relationships.
- Handles numerical and encoded categorical features.
- Does not require feature scaling in the same way as distance-based models.
- Can capture feature interactions.

### Limitations
- Deep trees can overfit.
- Small changes in data can produce a different tree.
- Greedy split selection may not find the globally best tree.
- A single tree may have lower predictive performance than an ensemble.
- Impurity-based feature importance can be misleading.

## 13. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Allow unrestricted depth | Tune max_depth and leaf settings |
| Select depth using test data repeatedly | Use validation or cross-validation |
| Treat feature importance as causation | Describe it as model-specific split contribution |
| Assume trees need standard scaling | Usually not necessary for ordinary decision trees |
| Ignore class imbalance | Inspect class metrics and consider class_weight |
| Interpret every branch as a business rule | Check stability and sample support |
| Use a very small leaf size without validation | Compare complexity and generalization |

## 14. Interview-ready explanations

**What is a Decision Tree?**  
A supervised algorithm that predicts by following a sequence of feature-based questions from a root node to a leaf.

**What is Gini impurity?**  
A measure of how mixed the classes are within a node. A pure node contains observations from one class.

**What is entropy?**  
A measure of uncertainty in the class distribution. A split can be selected to reduce uncertainty in child nodes.

**How do you prevent a tree from overfitting?**  
Control depth, minimum samples per split/leaf, leaf count or pruning, and validate the settings.

**Does a tree require feature scaling?**  
Ordinary decision trees are generally not sensitive to monotonic feature scaling because splits depend on feature ordering.

## 15. Scenario questions

1. Training accuracy is 100%, but test accuracy is much lower. What might be happening?  
   **Answer:** The tree may be too deep and overfitting training observations.

2. A tree is difficult to explain because it has hundreds of leaves. What can you do?  
   **Answer:** Limit depth, increase minimum leaf size or prune the tree, then validate the trade-off.

3. A feature has high impurity importance. Does it cause the target?  
   **Answer:** No. Importance describes how the fitted tree used the feature for splits, not causation.

4. A tree performs poorly on a minority class. What do you inspect?  
   **Answer:** Class distribution, confusion matrix, class weights, split strategy and feature representation.

5. Why might a tree be preferred over logistic regression for a first baseline?  
   **Answer:** If nonlinear feature interactions are important and a rule-based structure is useful, a shallow tree may be a reasonable candidate. Compare validation performance and interpretability.

## 16. Lesson completion checklist

- [ ] I can explain root, node, branch and leaf.
- [ ] I understand Gini and entropy conceptually.
- [ ] I can train a DecisionTreeClassifier.
- [ ] I can visualize a small tree.
- [ ] I can control depth and leaf size.
- [ ] I can inspect feature importance cautiously.
- [ ] I can train a DecisionTreeRegressor.
- [ ] I understand tree overfitting and pruning.

**Next lesson:** Day 30 — Random Forest and Ensemble Learning.
