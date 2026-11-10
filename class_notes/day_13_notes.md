# Day 13: Train, Validation and Test Split

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain why data is separated before model evaluation.
- Distinguish training, validation and test datasets.
- Use train_test_split with appropriate parameters.
- Explain random_state and reproducibility.
- Understand stratified splitting for classification.
- Recognize when group-based splitting is required.
- Recognize why time-series data needs chronological splitting.
- Check class proportions and row counts after splitting.
- Avoid using test data repeatedly to make model decisions.

## 2. Why do we split a dataset?

A machine-learning model must be evaluated on data that it did not use to learn its parameters. If the same observations are used for both fitting and evaluation, the measured performance may look better than the model's ability to handle new data.

Imagine a model that predicts whether a customer will churn. The team trains it on historical customer records and wants to estimate how it will perform on future customers.

A data split creates separate groups for different purposes.

## 3. Training, validation and test sets

| Dataset | Main purpose | How it is used |
|---|---|---|
| Training | Learn model parameters and preprocessing | Used during fit |
| Validation | Compare models, settings and thresholds | Used during development decisions |
| Test | Final evaluation after decisions are complete | Kept separate until final evaluation |

### Training set

The training set is used to fit the model. Preprocessing steps that learn from data, such as imputers and scalers, should also be fitted using training data only.

### Validation set

The validation set supports decisions during model development. It can be used to compare candidate models, tune hyperparameters and select a decision threshold.

### Test set

The test set is held back for a final evaluation. Repeatedly selecting models based on test performance turns the test set into another validation set and weakens the independence of the final result.

## 4. Create a small sample dataset

### 4.1. Import Pandas and NumPy

```python
import pandas as pd
import numpy as np

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

Pandas creates the example table and NumPy helps create reproducible demonstration data.



### 4.2. Create customer feature records

```python
df = pd.DataFrame({
    "customer_id": [f"C{i:03d}" for i in range(1, 21)],
    "age": [22, 35, 28, 41, 30, 45, 26, 38, 32, 50,
            24, 36, 29, 43, 31, 47, 27, 39, 33, 52],
    "monthly_spend": [1200, 4500, 2100, 5200, 3000, 6100, 1800, 4800, 3200, 7000,
                      1500, 4600, 2300, 5500, 3400, 6300, 1900, 4900, 3600, 7200],
    "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

print(df.head())
```

**Expected output**

```text
The first five customer records
```

**Explanation**

This is a small teaching dataset with a binary target. The customer ID is retained for traceability but excluded from model features in this example.



### 4.3 Select features and target

### 4.3. Create X and y

```python
X = df[["age", "monthly_spend"]]
y = df["churn"]

print(X.shape)
print(y.shape)
```

**Expected output**

```text
(20, 2) followed by (20,)
```

**Explanation**

X contains two feature columns and y contains one target value per row.



## 5. First split: training and test

The simplest common split creates a training set and a test set. The test set is held back for final evaluation.

### 5.1 Import train_test_split

### 5.1. Import the splitting function

```python
from sklearn.model_selection import train_test_split

print("train_test_split imported")
```

**Expected output**

```text
train_test_split imported
```

**Explanation**

The function divides aligned features and target values into subsets.



### 5.2 Split features and target together

### 5.2. Create a train-test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

**Expected output**

```text
(16, 2), (4, 2), (16,), (4,)
```

**Explanation**

The test_size value reserves 20% of the observations for testing. The function splits X and y together so their rows remain paired.



### 5.3 Understand test_size

- `test_size=0.2` reserves 20% for testing.
- `test_size=0.3` reserves 30%.
- The remaining observations are used for training when `train_size` is not separately specified.

### 5.3. Try a 30 percent test set

```python
X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print(len(X_train_30))
print(len(X_test_30))
```

**Expected output**

```text
14 followed by 6
```

**Explanation**

The test set is larger, leaving fewer rows for training. Choose the split based on dataset size, evaluation needs and the task.



### 5.4 Understand random_state

A random split can produce different records each time. Setting `random_state` makes the split repeatable for the same data and software behaviour.

### 5.4. Repeat a split with the same random state

```python
split_a = train_test_split(
    X, y, test_size=0.2, random_state=42
)

split_b = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(split_a[1].index.equals(split_b[1].index))
```

**Expected output**

```text
True
```

**Explanation**

The same input and random_state reproduce the same split. Reproducibility also depends on data version, code and library behaviour.



### 5.5 Compare splits with different random states

### 5.5. Use a different random state

```python
split_c = train_test_split(
    X, y, test_size=0.2, random_state=7
)

print(split_a[1].index.tolist())
print(split_c[1].index.tolist())
```

**Expected output**

```text
Two test-index lists that may differ
```

**Explanation**

A different seed can select different records. Do not repeatedly try seeds until a preferred test score appears.



## 6. Stratified splitting for classification

In classification, the target may contain different class proportions. If the positive class is rare, a random split may leave too few examples of that class in one subset.

**Stratification** attempts to preserve the approximate class proportions across the split.

### 6.1 Inspect target distribution

### 6.1. Count target labels

```python
print(y.value_counts())
```

**Expected output**

```text
Counts for class 0 and class 1
```

**Explanation**

Understand the class distribution before choosing a split strategy.



### 6.2 Calculate class proportions

### 6.2. Inspect target proportions

```python
print(y.value_counts(normalize=True))
```

**Expected output**

```text
Approximate proportions of the two classes
```

**Explanation**

A stratified split aims to preserve these proportions, subject to integer rounding and class-size constraints.



### 6.3 Use stratify=y

### 6.3. Create a stratified split

```python
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(y_train_s.value_counts(normalize=True).sort_index())
print(y_test_s.value_counts(normalize=True).sort_index())
```

**Expected output**

```text
Similar class proportions in train and test
```

**Explanation**

The stratify argument uses the target labels to keep class representation approximately consistent. It is commonly useful for classification.



### 6.4 Understand when stratification can fail

Stratification may fail when a class has too few observations to place in both subsets or when the requested split is too small for all classes.

**Developer habit:** Check the number of examples in each class before splitting. Do not force stratification if the dataset cannot support it; revisit the data collection or evaluation plan.

### 6.5 Regression tasks

For ordinary regression, the target is continuous rather than a set of classes. Standard stratification on raw target values is usually not directly appropriate. A team may use carefully designed bins for approximate distribution preservation in some cases, but the strategy must be justified and should not distort the evaluation.

## 7. Create training, validation and test sets

A three-way split is useful when a separate validation set is needed for development decisions.

### 7.1 First separate test data

### 7.1. Hold out a final test set

```python
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_dev.shape, X_test.shape)
```

**Expected output**

```text
Development data and test data shapes
```

**Explanation**

The test set is set aside. The development data will be divided into training and validation sets.



### 7.2 Split development data into training and validation

### 7.2. Create training and validation subsets

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
```

**Expected output**

```text
12 training rows, 4 validation rows and 4 test rows (for this 20-row example)
```

**Explanation**

The validation fraction is 25% of the 16-row development set. This produces 60% training, 20% validation and 20% test overall.



### 7.3 Compare class proportions

### 7.3. Check all three target subsets

```python
for name, labels in [
    ("train", y_train),
    ("validation", y_valid),
    ("test", y_test)
]:
    print(name)
    print(labels.value_counts(normalize=True).sort_index())
```

**Expected output**

```text
A class-proportion summary for training, validation and test
```

**Explanation**

The subsets should have broadly similar class proportions when stratification is feasible. Small datasets can still show rounding differences.



### 7.4 Why keep the test set separate?

Use the validation set to make development decisions. Once the model design and settings are finalized, evaluate on the test set. If you repeatedly tune based on test results, the test set no longer provides an independent final check.

## 8. Group-based splitting

Some datasets contain multiple records from the same entity. If the deployment goal is to predict for unseen entities, records from one entity should not appear in both training and test subsets.

Examples:
- Multiple transactions per customer.
- Multiple scans per patient.
- Multiple measurements per machine.
- Multiple sessions per user.

### 8.1 Create repeated customer observations

### 8.1. Build a grouped dataset

```python
group_df = pd.DataFrame({
    "customer_id": ["A", "A", "A", "B", "B", "B", "C", "C", "D", "D", "E", "E"],
    "month": [1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2],
    "usage": [10, 12, 11, 5, 6, 4, 15, 14, 8, 9, 7, 6],
    "churn_next_month": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1]
})

print(group_df)
```

**Expected output**

```text
Twelve customer-month rows belonging to five customers
```

**Explanation**

The same customer appears in multiple rows. A random row split could put one customer's history into both subsets.



### 8.2 Split by group using GroupShuffleSplit

### 8.2. Create a group-based split

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.3,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        group_df[["month", "usage"]],
        group_df["churn_next_month"],
        groups=group_df["customer_id"]
    )
)

group_train = group_df.iloc[train_idx]
group_test = group_df.iloc[test_idx]

print(group_train["customer_id"].unique())
print(group_test["customer_id"].unique())
```

**Expected output**

```text
Two sets of customer IDs with no overlap
```

**Explanation**

GroupShuffleSplit keeps each group entirely in one subset. The exact selected groups depend on the split and seed.



### 8.3 Check for group overlap

### 8.3. Validate no customer appears in both subsets

```python
train_customers = set(group_train["customer_id"])
test_customers = set(group_test["customer_id"])

print(train_customers.intersection(test_customers))
```

**Expected output**

```text
set()
```

**Explanation**

An empty intersection means no customer ID is shared across these two subsets.



### 8.4 When should group splitting be used?

Use group splitting when the evaluation goal is generalization to new entities. If the production task is to predict future events for known customers, the split may need to preserve time order as well. Choose based on the real prediction scenario.

## 9. Time-based splitting

For time-dependent data, random splitting can allow future records into training while earlier records are placed in testing. This may not reflect how the model will be used.

A chronological split trains on earlier data and evaluates on later data.

### 9.1 Create time-ordered observations

### 9.1. Create a monthly sales dataset

```python
time_df = pd.DataFrame({
    "month": pd.date_range("2026-01-01", periods=10, freq="MS"),
    "marketing_spend": [100, 110, 105, 120, 130, 125, 140, 150, 145, 160],
    "sales": [500, 520, 510, 560, 590, 580, 620, 650, 640, 680]
})

print(time_df)
```

**Expected output**

```text
Ten monthly observations in chronological order
```

**Explanation**

Each row represents one month. The model should use earlier information to predict a later period.



### 9.2 Sort by time

### 9.2. Ensure chronological order

```python
time_df = time_df.sort_values("month")

print(time_df["month"])
```

**Expected output**

```text
The month values in ascending chronological order
```

**Explanation**

Sort explicitly rather than assuming the source file is already ordered.



### 9.3 Split earlier and later periods

### 9.3. Use the first eight months for training

```python
time_train = time_df.iloc[:8]
time_test = time_df.iloc[8:]

print(time_train["month"].min(), time_train["month"].max())
print(time_test["month"].min(), time_test["month"].max())
```

**Expected output**

```text
Training covers the earlier months; test covers the final two months
```

**Explanation**

The test period occurs after the training period. This better reflects forecasting into the future than a random row split.



### 9.4 Check the time boundary

### 9.4. Validate chronological separation

```python
print(time_train["month"].max() < time_test["month"].min())
```

**Expected output**

```text
True
```

**Explanation**

The latest training month is earlier than the earliest test month. More complex time-series evaluation may use rolling or expanding windows.



## 10. Understand leakage through an incorrect split

### Random split on repeated entities

If the same customer has multiple records, random row splitting may place some of those records in both training and test data. The model may recognize customer-specific patterns rather than generalize to new customers.

### Random split on time-series data

A random split may train on future months and test on earlier months. This can make forecasting performance unrealistically optimistic.

### Preprocessing before splitting

If a scaler or imputer is fitted on all observations before the split, the held-out data influences learned preprocessing statistics.

## 11. Verify a split

After splitting, check:
- Row counts.
- Feature and target lengths.
- Class proportions for classification.
- Group overlap for grouped data.
- Time boundaries for chronological data.
- Duplicate identifiers where relevant.

### 11.1 Check feature-target alignment

### 11.1. Compare row counts

```python
print(len(X_train), len(y_train))
print(len(X_valid), len(y_valid))
print(len(X_test), len(y_test))
```

**Expected output**

```text
Matching feature and target row counts for each subset
```

**Explanation**

train_test_split splits X and y together. Always verify the resulting shapes.



### 11.2 Check index overlap for ordinary random split

### 11.2. Check that train and test indices do not overlap

```python
print(set(X_train.index).intersection(set(X_test.index)))
```

**Expected output**

```text
set()
```

**Explanation**

For this ordinary random split, an empty intersection confirms that the same original row indices were not selected in both subsets.



### 11.3 Check duplicate customer entities where appropriate

### 11.3. Compare customer IDs for group split

```python
print(
    set(group_train["customer_id"]).intersection(
        set(group_test["customer_id"])
    )
)
```

**Expected output**

```text
set()
```

**Explanation**

For a new-customer evaluation, no entity should occur in both sets. This check is not required for every possible prediction design; choose the rule based on the task.



## 12. Choosing a split strategy

| Data situation | Possible split strategy | Why |
|---|---|---|
| Independent rows | Random train-test split | Rows can be treated as independent observations |
| Classification with imbalanced classes | Stratified split | Preserves approximate class proportions |
| Multiple records per entity, unseen-entity goal | Group split | Keeps entities separate |
| Forecasting future periods | Chronological split | Prevents future-to-past leakage |
| Repeated measurements over time and entities | Time-aware group strategy | May need to preserve both entity and time boundaries |

No single split strategy is correct for every dataset. It must represent the deployment question.

## 13. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Train and test on the same records | Evaluation is not independent | Hold out unseen observations |
| Tune repeatedly on test data | Test set becomes part of development | Use validation data for decisions |
| Forget stratify for imbalanced classification | A small subset may miss a class | Inspect class counts and stratify when suitable |
| Randomly split time-series rows | Future information can leak into training | Use chronological splits |
| Randomly split repeated customer rows | Same entity can occur in both sets | Use group splitting when required |
| Fit scaler before splitting | Held-out information influences preprocessing | Split first and fit on training only |
| Choose split percentage without checking data size | Validation/test may be too small | Consider sample size and evaluation uncertainty |
| Use random_state as a quality setting | It only controls reproducibility of a random split | Choose split design first, then set a seed |

## 14. Interview-ready explanations

**Why do we split data into training and test sets?**  
Training data is used to fit the model. Test data is held back to estimate how well the model performs on unseen observations.

**What is the purpose of a validation set?**  
It supports model selection, hyperparameter tuning and threshold decisions during development. The final test set should remain separate until those decisions are complete.

**What is stratified splitting?**  
It attempts to preserve the target class proportions in each subset. It is useful for classification, especially when classes are imbalanced and the dataset has enough examples.

**When should you use group splitting?**  
When multiple observations belong to the same entity and the evaluation goal requires testing on unseen entities. It prevents the same group from appearing in both training and test.

**Why use chronological splitting for time series?**  
The model should learn from earlier periods and be evaluated on later periods, reflecting the direction of real forecasting.

## 15. Scenario questions

1. A churn dataset has 95% non-churn and 5% churn. What should you consider when splitting?  
   **Answer:** Inspect class counts and use stratification if the sample supports it. Later evaluate with metrics appropriate to the minority class.

2. Each patient has ten medical records. The goal is to predict outcomes for new patients. What split is appropriate?  
   **Answer:** Group-based splitting by patient ID, so one patient's records do not appear in both training and test.

3. You are forecasting next month's sales. Can you randomly shuffle all months into train and test?  
   **Answer:** Not for a realistic future forecast. Use chronological or time-series validation.

4. The team selects the model with the highest test score after trying 50 models. Is the test score still an independent final estimate?  
   **Answer:** No. The test set has influenced model selection. Use validation for decisions and reserve a new independent test set if possible.

5. The test set has no positive examples after a random split. What do you do?  
   **Answer:** Investigate class counts and split design. Use stratification where appropriate, or collect more data if there are too few positive examples.

## 16. Lesson completion checklist

- [ ] I can explain training, validation and test sets.
- [ ] I can use train_test_split.
- [ ] I understand test_size and random_state.
- [ ] I can explain stratification.
- [ ] I can identify when group splitting is needed.
- [ ] I can create a chronological split.
- [ ] I can check row counts and target alignment.
- [ ] I understand why test data should not drive repeated tuning.
- [ ] I can select a split strategy based on deployment.

**Next lesson:** Day 14 — EDA, Distributions and Feature Relationships.
