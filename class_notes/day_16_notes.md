# Day 16: Data Leakage and Preprocessing Pipelines

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define target leakage and train-test contamination.
- Identify features unavailable at prediction time.
- Explain why preprocessing must be fitted on training data only.
- Use scikit-learn Pipeline.
- Design a leakage-safe evaluation workflow.

## 1. What Is Data Leakage?
Data leakage occurs when training or evaluation uses information that would not legitimately be available when the real prediction is made, or when held-out data influences training decisions.

### Example types
1. Target leakage: a feature is created after the outcome.
2. Preprocessing leakage: scaler/imputer is fitted on all rows before splitting.
3. Entity leakage: records for the same customer appear in both sets when testing unseen customers.

## 2. Pipeline Syntax
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```
The pipeline fits the scaler and model using training data and applies the learned transformation during prediction.

## 3. Three Leakage Examples
1. Predict churn but include cancellation_reason entered after cancellation.
2. Impute missing values using the full dataset before train-test split.
3. Randomly split customer-month rows so the same customer is in both sets.

## 4. Scenario Questions
1. A hospital model predicts readmission but uses a field recorded after readmission. What is wrong? Target leakage.
2. A scaler is fitted before splitting. What is wrong? Test distribution influenced preprocessing.
3. Offline score is suspiciously close to 100%. What do you investigate? Target leakage, duplicates, split logic, timestamps and feature availability.
