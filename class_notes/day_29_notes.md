# Day 29: Decision Trees, Splits and Impurity

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define a decision tree.
- Explain root, node, branch and leaf.
- Understand how classification trees select splits.
- Recognize Gini and entropy.
- Control overfitting.

## 1. Definition
A decision tree makes predictions by repeatedly splitting data using feature conditions. A path from the root to a leaf represents a sequence of decisions.

## 2. Important Terms
- Root: first decision node.
- Internal node: a decision within the tree.
- Branch: outcome of a condition.
- Leaf: final prediction.
- Impurity: measure of how mixed class labels are in a node.

## 3. Syntax
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 4. Three Examples
1. Loan risk decision based on debt-to-income and repayment history.
2. Predict customer churn from usage and support contacts.
3. Classify flower species from measurements.

## 5. Scenario Questions
1. Training score is 100%, test score is low. What might be happening? Tree overfitting.
2. Why limit max_depth? To restrict complexity.
3. Does feature importance prove causation? No; it describes model split usage, not causal effect.
