# Day 13: Train, Validation and Test Split

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain why data must be separated for training and evaluation.
- Distinguish training, validation and test sets.
- Use train_test_split and stratification.
- Recognize group leakage and time leakage.
- Select a split strategy based on the real prediction task.

## 1. What Is a Data Split?
A data split separates observations so we can train a model and estimate how it performs on examples it has not learned from.

- Training set: fits model parameters and preprocessing.
- Validation set: supports model/parameter/threshold selection.
- Test set: final evaluation after decisions are complete.

## 2. Syntax
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```
- X: features; y: target.
- test_size: fraction reserved for test.
- random_state: repeatable random split.
- stratify: preserve approximate class proportions for classification.

## 3. Three Split Examples
1. Independent observations: random train-test split.
2. Imbalanced classification: stratified split so rare class proportions are represented.
3. Time series: chronological split so future records are not used to predict the past.

## 4. Group-Based Splitting
When multiple rows belong to the same customer, patient or device, keep groups separated if the deployment goal is performance on unseen entities. Otherwise, the same entity can appear in both train and test.

## 5. Scenario Questions
1. Same customer appears in train and test. What is the risk? Group leakage and optimistic performance.
2. You predict next month's sales. Should you randomly mix future and past rows? No; preserve time order.
3. You tune parameters repeatedly using the test score. Is the test set still independent? No; it has influenced model selection.
