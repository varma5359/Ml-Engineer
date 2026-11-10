# Day 26: Decision Boundaries and Classification Thresholds

**Pathway:** Machine Learning Engineer  
**Phase:** 3 — Supervised learning  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define a decision boundary.
- Explain how probability thresholds assign classes.
- Understand the trade-off between precision and recall.
- Select a threshold using validation data.

## 1. Definition
A decision boundary separates regions of feature space that a classifier assigns to different classes. A threshold converts a model score or probability into a final class label.

## 2. Threshold Syntax
```python
probabilities = [0.20, 0.45, 0.80]
threshold = 0.50
predictions = [int(p >= threshold) for p in probabilities]
print(predictions)
```
Output: [0, 0, 1].

## 3. Three Threshold Examples
1. High threshold: fewer positive alerts, often higher precision but lower recall.
2. Low threshold: more positive alerts, often higher recall but lower precision.
3. Threshold chosen under review capacity: choose a threshold that matches how many cases the team can investigate.

## 4. Scenario Questions
1. Fraud is expensive to miss. Which metric deserves attention? Recall, alongside precision and review cost.
2. Team can review only 100 alerts per day. What must threshold selection consider? Alert volume and operational capacity.
3. Can you select the final threshold on the test set? No; use validation data and keep test data for final evaluation.
