# Day 25: Logistic Regression and Sigmoid Function

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain why logistic regression is a classification algorithm.
- Understand linear score and sigmoid probability.
- Train a binary classifier.
- Use predict() and predict_proba().
- Recognize threshold and class-balance issues.

## 1. Definition
Logistic regression is a supervised classification algorithm that estimates class probability by applying a logistic (sigmoid) function to a linear score. Despite the name, it is commonly used for classification.

## 2. Sigmoid
The sigmoid function converts any real-valued score into a value between 0 and 1, which can be interpreted as an estimated positive-class probability when the model is appropriate.

## 3. Syntax
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predicted_classes = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## 4. Three Examples
1. Predict customer churn: yes/no.
2. Predict whether an email is spam: spam/not spam.
3. Predict whether a transaction is flagged for review: yes/no.

## 5. Scenario Questions
1. Model probability is 0.82. Does it guarantee the customer will churn? No; it is an estimated probability.
2. Model predicts every customer as non-churn. What do you inspect? Class balance, threshold, recall and confusion matrix.
3. Why might max_iter be increased? To allow the optimizer more iterations to converge; also inspect scaling and warnings.
