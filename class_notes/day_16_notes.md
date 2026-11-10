# Day 16: Data Leakage and Preprocessing Pipelines

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain data leakage in plain language.
- Distinguish target leakage, preprocessing leakage, entity leakage and time leakage.
- Identify features unavailable at prediction time.
- Explain why preprocessing must be fitted on training data only.
- Use train-test splitting before learned transformations.
- Build a scikit-learn Pipeline.
- Combine numeric and categorical transformations using ColumnTransformer.
- Evaluate a pipeline without manually transforming the test set.
- Recognize common leakage mistakes in real ML projects.

## 2. What is data leakage?

**Data leakage** happens when information that should not be available to the model during training or evaluation influences the learning process or prediction.

The model may show excellent evaluation results but fail when deployed because the evaluation used information that would not exist at the real prediction time.

Think of a student taking an exam after seeing the answer sheet. The score does not show whether the student can solve a new exam independently.

### Common leakage types

| Type | Meaning | Example |
|---|---|---|
| Target leakage | A feature contains information created after or from the outcome | Cancellation reason used to predict cancellation |
| Preprocessing leakage | A transformation learns from validation/test data | Scaler fitted on the complete dataset |
| Entity leakage | Related records from the same entity appear in train and test | Same patient in both sets when evaluating unseen patients |
| Time leakage | Future records influence training for a past-to-future task | Randomly splitting monthly sales |
| Selection leakage | Test results repeatedly guide model selection | Choosing among many models using the final test set |

## 3. Create a practice dataset

### 3.1 Import libraries

### 3.1. Import Pandas and NumPy

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

Pandas and NumPy are used to create and inspect the example data.



### 3.2 Create a customer dataset with missing values

### 3.2. Build the example dataset

```python
df = pd.DataFrame({
    "age": [25, 34, 29, 42, 31, 45, 27, 39, 33, 50, 24, 36],
    "monthly_spend": [4500, 5200, 6100, np.nan, 4900, 7000, 1900, 5800, 3600, 8200, 1600, 5400],
    "city": ["Hyderabad", "Chennai", "Vizag", "Hyderabad", "Chennai", "Vizag",
             "Hyderabad", "Chennai", "Vizag", "Hyderabad", "Chennai", None],
    "contract_type": ["Monthly", "Annual", "Monthly", "Annual", "Monthly", "Annual",
                      "Monthly", "Annual", "Monthly", "Annual", "Monthly", "Annual"],
    "churn": [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
})

print(df.head())
```

**Expected output**

```text
The first five rows with numeric and categorical fields and some missing values
```

**Explanation**

The target is churn. The features contain two numeric and two categorical columns. This is a small demonstration dataset.



### 3.3 Separate features and target

### 3.3. Create X and y

```python
X = df.drop(columns=["churn"])
y = df["churn"]

print(X.shape)
print(y.shape)
```

**Expected output**

```text
(12, 4) followed by (12,)
```

**Explanation**

The target is separated from the model inputs. This prevents accidentally including the target as a feature.



## 4. Target leakage

Target leakage occurs when an input feature contains information about the target that would not be available at prediction time.

### Example: cancellation prediction

Suppose the business wants to predict whether a customer will cancel next month.

Potential features:
- Customer tenure today.
- Usage in the previous month.
- Support calls up to today.
- Cancellation reason recorded after cancellation.

Cancellation reason may directly reveal the outcome and is not available before cancellation.

### 4.1 Create a leakage example

### 4.1. Add a post-outcome field

```python
leakage_df = pd.DataFrame({
    "usage_last_month": [10, 12, 8, 20, 18, 7],
    "support_calls": [2, 1, 4, 0, 1, 5],
    "cancellation_reason": ["Price", None, None, None, None, "Service"],
    "churn": [1, 0, 1, 0, 0, 1]
})

print(leakage_df)
```

**Expected output**

```text
A table where cancellation_reason is populated for churned customers
```

**Explanation**

This field is a post-outcome record in this scenario. Its strong relationship with churn does not make it a valid prediction-time feature.



### 4.2 Inspect target and leakage field together

### 4.2. Compare cancellation reason and churn

```python
print(pd.crosstab(
    leakage_df["cancellation_reason"].notna(),
    leakage_df["churn"]
))
```

**Expected output**

```text
A cross-tabulation showing the relationship between whether a cancellation reason exists and the churn label
```

**Explanation**

This reveals that the field's presence is closely connected to the outcome. The timestamp and business definition establish whether it is leakage.



### 4.3 Exclude the post-outcome field

### 4.3. Select only pre-outcome inputs

```python
safe_features = leakage_df[
    ["usage_last_month", "support_calls"]
]

print(safe_features)
```

**Expected output**

```text
A feature table without cancellation_reason
```

**Explanation**

This selection matches the scenario's prediction-time constraint. Always verify actual data availability rather than relying on column names alone.



## 5. Preprocessing leakage

Many preprocessing operations learn information from the supplied dataset:
- Imputation learns a mean or median.
- Scaling learns means, standard deviations, minima or maxima.
- Encoding learns which categories exist.
- Feature selection learns which features appear useful.
- Dimensionality reduction learns components from the data.

If these operations are fitted on the complete dataset before splitting, held-out information influences the transformation.

### 5.1 Create a numeric example

### 5.1. Create a numerical feature

```python
values = pd.DataFrame({
    "income": [20000, 25000, 30000, 35000, 40000, 100000]
})

print(values)
```

**Expected output**

```text
Six income values
```

**Explanation**

The final value is much higher than the others. We will use the data to demonstrate why a scaler must be fitted on training data only.



### 5.2 Demonstrate an incorrect fit on all data

### 5.2. Fit a scaler before splitting (incorrect)

```python
from sklearn.preprocessing import StandardScaler

wrong_scaler = StandardScaler()
all_scaled = wrong_scaler.fit_transform(values)

print(wrong_scaler.mean_)
```

**Expected output**

```text
The mean learned from all six observations
```

**Explanation**

This is not a valid evaluation workflow because the held-out observations contributed to the learned mean.



### 5.3 Split before fitting

### 5.3. Create training and test data first

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(
    values,
    test_size=0.33,
    random_state=42
)

print(X_train)
print(X_test)
```

**Expected output**

```text
Separate training and test rows
```

**Explanation**

The test rows are held out before learning preprocessing statistics.



### 5.4 Fit the scaler on training data only

### 5.4. Fit on training data

```python
correct_scaler = StandardScaler()

X_train_scaled = correct_scaler.fit_transform(X_train)

print(correct_scaler.mean_)
```

**Expected output**

```text
A mean calculated from X_train only
```

**Explanation**

The scaler learns from training data. This is the correct order for an unbiased held-out evaluation.



### 5.5 Transform test data using the fitted scaler

### 5.5. Transform test data

```python
X_test_scaled = correct_scaler.transform(X_test)

print(X_test_scaled)
```

**Expected output**

```text
Test values transformed using the training mean and standard deviation
```

**Explanation**

Do not call fit or fit_transform on the test data. Apply the transformation learned from training.



## 6. What is a preprocessing pipeline?

A **pipeline** is an ordered sequence of data transformations and an estimator. It ensures that the same preprocessing steps are applied during model training and prediction.

Benefits:
- Reduces inconsistent manual preprocessing.
- Helps prevent leakage when used correctly with model selection.
- Makes the workflow easier to reproduce.
- Keeps transformation and model logic together.
- Simplifies prediction on new records.

## 7. Build a basic pipeline

### 7.1 Create a numeric dataset

### 7.1. Prepare feature and target values

```python
X_numeric = pd.DataFrame({
    "age": [20, 25, 30, 35, 40, 45, 50, 55, 28, 38, 48, 58],
    "monthly_spend": [1000, 1500, 2200, 3000, 3500, 4200, 5000, 5800, 1800, 3200, 4500, 6200]
})

y_numeric = pd.Series([0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1])

print(X_numeric.head())
```

**Expected output**

```text
A numeric feature table
```

**Explanation**

The values are for learning only. A production target must be defined and collected independently.



### 7.2 Split the dataset

### 7.2. Split before preprocessing

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric,
    y_numeric,
    test_size=0.25,
    random_state=42,
    stratify=y_numeric
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
Training and test feature shapes
```

**Explanation**

The feature and target rows are split together. Stratification preserves approximate class proportions when the sample supports it.



### 7.3 Create a scaler

### 7.3. Initialize StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print(scaler)
```

**Expected output**

```text
A StandardScaler instance
```

**Explanation**

Initialization defines the transformation object. It has not learned any data statistics yet.



### 7.4 Create a model

### 7.4. Initialize logistic regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

print(model)
```

**Expected output**

```text
A LogisticRegression instance
```

**Explanation**

The model is initialized but not trained. max_iter sets the iteration limit for the optimization routine.



### 7.5 Create a Pipeline

### 7.5. Connect scaler and model

```python
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

print(numeric_pipeline)
```

**Expected output**

```text
A Pipeline containing scale and model steps
```

**Explanation**

The steps run in order. During fit, the scaler is fitted on training inputs and the transformed data is passed to logistic regression.



### 7.6 Fit the pipeline

### 7.6. Train the complete pipeline

```python
numeric_pipeline.fit(X_train, y_train)

print("Training complete")
```

**Expected output**

```text
Training complete
```

**Explanation**

Calling fit on the pipeline fits each transformer and then the final estimator using the training data.



### 7.7 Predict test observations

### 7.7. Predict with the fitted pipeline

```python
predictions = numeric_pipeline.predict(X_test)

print(predictions)
```

**Expected output**

```text
Predicted class labels for the held-out observations
```

**Explanation**

The pipeline scales test data using the training-fitted scaler and passes the result to the model.



### 7.8 Calculate a basic accuracy score

### 7.8. Evaluate test predictions

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
```

**Expected output**

```text
An accuracy value between zero and one
```

**Explanation**

This is a small demonstration dataset. Do not treat its score as evidence of production readiness. Later lessons cover evaluation metrics and limitations.



## 8. Mixed numerical and categorical preprocessing

Real datasets contain different column types. A numeric column may need imputation and scaling; a categorical column may need imputation and one-hot encoding.

### 8.1 Create a mixed-type dataset

### 8.1. Build mixed features with missing values

```python
mixed_X = pd.DataFrame({
    "age": [22, 35, 28, 41, 30, 46, 25, 39, 33, 50, 27, 43],
    "monthly_spend": [1200, 4500, np.nan, 5200, 3000, 6100, 1800, 4800, 3200, 7000, 1900, 5500],
    "city": ["Hyd", "Chennai", "Hyd", "Vizag", None, "Chennai", "Hyd", "Vizag", "Chennai", "Hyd", "Vizag", "Chennai"],
    "contract": ["Monthly", "Annual", "Monthly", "Annual", "Monthly", "Annual",
                 "Monthly", "Annual", "Monthly", "Annual", "Monthly", "Annual"]
})

mixed_y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

print(mixed_X.head())
```

**Expected output**

```text
A mixed numerical/categorical feature table with missing values
```

**Explanation**

The example contains numeric and categorical features. The target is separate.



### 8.2 Identify column groups

### 8.2. Create numeric and categorical column lists

```python
numeric_features = ["age", "monthly_spend"]
categorical_features = ["city", "contract"]

print(numeric_features)
print(categorical_features)
```

**Expected output**

```text
Two feature-name lists
```

**Explanation**

Explicit column groups make the preprocessing design easier to review and maintain.



### 8.3 Split before fitting the preprocessing objects

### 8.3. Split mixed data and target

```python
X_train, X_test, y_train, y_test = train_test_split(
    mixed_X,
    mixed_y,
    test_size=0.25,
    random_state=42,
    stratify=mixed_y
)

print(X_train.shape, X_test.shape)
```

**Expected output**

```text
Training and test shapes
```

**Explanation**

All learned preprocessing will be fitted inside the pipeline using X_train only.



### 8.4 Build a numeric transformer

### 8.4. Impute and scale numerical columns

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

print(numeric_transformer)
```

**Expected output**

```text
A numeric pipeline containing imputation and scaling
```

**Explanation**

The median is learned during fit on training data. The scaler is also fitted on the imputed training values.



### 8.5 Build a categorical transformer

### 8.5. Impute and one-hot encode categories

```python
from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

print(categorical_transformer)
```

**Expected output**

```text
A categorical pipeline containing imputation and one-hot encoding
```

**Explanation**

The most frequent category is learned from training data. Unknown categories at transform time are handled by the encoder configuration.



### 8.6 Combine with ColumnTransformer

### 8.6. Create a combined preprocessor

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

print(preprocessor)
```

**Expected output**

```text
A ColumnTransformer with numeric and categorical branches
```

**Explanation**

Each branch applies only to its selected columns. The result is a combined transformed feature matrix.



### 8.7 Connect preprocessing to a model

### 8.7. Create the full ML pipeline

```python
mixed_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print(mixed_pipeline)
```

**Expected output**

```text
A pipeline with preprocessing and classifier steps
```

**Explanation**

The full pipeline contains every learned preprocessing step and the estimator.



### 8.8 Fit the pipeline

### 8.8. Fit using training data only

```python
mixed_pipeline.fit(X_train, y_train)

print("Mixed-data pipeline fitted")
```

**Expected output**

```text
Mixed-data pipeline fitted
```

**Explanation**

The imputer, scaler, encoder and classifier learn from the training partition only.



### 8.9 Predict held-out rows

### 8.9. Predict on test data

```python
mixed_predictions = mixed_pipeline.predict(X_test)

print(mixed_predictions)
```

**Expected output**

```text
Predicted labels for test observations
```

**Explanation**

The pipeline applies exactly the fitted preprocessing sequence to test records.



### 8.10 Inspect transformed feature names

### 8.10. Read output feature names

```python
feature_names = mixed_pipeline.named_steps[
    "preprocessing"
].get_feature_names_out()

print(feature_names)
```

**Expected output**

```text
Names for numeric and one-hot encoded output columns
```

**Explanation**

Feature names help explain the model's transformed inputs. The names include transformer prefixes.



## 9. Cross-validation and pipelines

Cross-validation repeatedly trains and validates a model using different folds. A pipeline ensures preprocessing is fitted separately within each training fold rather than once on all observations.

### 9.1 Import cross-validation tools

### 9.1. Import cross_val_score

```python
from sklearn.model_selection import cross_val_score

print("Cross-validation imported")
```

**Expected output**

```text
Cross-validation imported
```

**Explanation**

Cross-validation evaluates a workflow over multiple data partitions.



### 9.2 Evaluate the pipeline with cross-validation

### 9.2. Run cross-validation on the full development data

```python
cv_scores = cross_val_score(
    mixed_pipeline,
    mixed_X,
    mixed_y,
    cv=3,
    scoring="accuracy"
)

print(cv_scores)
```

**Expected output**

```text
Three fold accuracy values
```

**Explanation**

The estimator is cloned and fitted within each fold. With a tiny dataset, results are highly unstable and only demonstrate the workflow.



### 9.3 Inspect average and variation

### 9.3. Summarize cross-validation scores

```python
print("Mean:", cv_scores.mean())
print("Std:", cv_scores.std())
```

**Expected output**

```text
The average fold score and its standard deviation
```

**Explanation**

The mean summarizes the folds; the standard deviation indicates how much fold scores vary. Neither is a guarantee of production performance.



## 10. Common leakage patterns

### Pattern 1: Impute before splitting

Incorrect:
```python
df["age"] = df["age"].fillna(df["age"].median())
X_train, X_test = train_test_split(df)
```

The median was learned from all rows, including test observations.

Better: split first and use a fitted imputer inside a pipeline.

### Pattern 2: Scale before splitting

Incorrect:
```python
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

The scaler learned from test rows.

Better: fit the scaler on training data only.

### Pattern 3: Use a future field

A delivery prediction system uses actual_delivery_time to predict whether a delivery will be late. The field is known only after delivery, so it leaks the outcome.

### Pattern 4: Random split repeated entities

A customer-month dataset is randomly split by row even though the intended evaluation is for unseen customers. The same customer appears in both sets.

### Pattern 5: Tune on the test set

A team tries many model settings and chooses the one with the highest test score. The test data has now influenced development decisions.

## 11. A practical leakage review checklist

Before accepting an ML training workflow, ask:

1. What is the prediction timestamp?
2. Is every feature available at that timestamp?
3. Is the target derived from future outcomes?
4. Are any columns direct copies or proxies of the target?
5. Were imputers, scalers and encoders fitted only on training data?
6. Are related entities separated if required?
7. Is time order preserved for forecasting?
8. Has the test set been kept out of model selection?
9. Does inference use the same pipeline as training?
10. Are all preprocessing choices recorded?

## 12. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Use a post-outcome field | Model sees information unavailable in production | Check timestamps and definitions |
| Fit preprocessing before split | Held-out data influences learned statistics | Split first and use Pipeline |
| Transform test data with a new scaler | Training and test representations differ | Call transform on the fitted scaler |
| Preprocess train and test separately | Inconsistent categories/scales | Use one fitted preprocessing pipeline |
| Ignore entity/time structure | Related or future observations leak | Use group/time-aware splitting |
| Use test scores to tune repeatedly | Test set becomes part of development | Use validation or cross-validation |
| Manually duplicate preprocessing code | Training/inference can diverge | Serialize and reuse the pipeline |
| Treat high performance as proof of quality | Leakage can inflate scores | Audit feature availability and split design |

## 13. Interview-ready explanations

**What is data leakage?**  
Data leakage occurs when training or evaluation uses information that would not legitimately be available when the prediction is made, or when held-out data influences model development.

**What is target leakage?**  
A feature contains information created after or derived from the target event. It can make model performance look unrealistically high.

**Why fit preprocessing on training data only?**  
Imputers, scalers and encoders learn information from data. Fitting them on the full dataset lets validation/test information influence the transformation.

**What is a Pipeline?**  
A Pipeline chains preprocessing steps and an estimator so they are fitted and applied in a consistent order.

**Why use ColumnTransformer?**  
It applies different preprocessing to selected column groups, such as scaling numerical features and encoding categorical features.

## 14. Scenario questions

1. A churn model uses cancellation_reason. What do you investigate?  
   **Answer:** When the field is recorded. If it is entered after cancellation, it is target leakage.

2. The scaler is fitted on all rows before train-test split. What is wrong?  
   **Answer:** Test information influenced the learned scaling statistics. Split first and fit only on training data.

3. A model scores 99% accuracy but performs poorly in production. What do you investigate?  
   **Answer:** Leakage, train/production preprocessing differences, data shift, feature availability and evaluation design.

4. A new city appears in production. What should happen?  
   **Answer:** The encoder should have an explicit unknown-category policy and the system should monitor new categories.

5. The same patient appears in train and test, but the goal is unseen-patient prediction. What is wrong?  
   **Answer:** Entity leakage. Use patient-level group splitting.

## 15. Lesson completion checklist

- [ ] I can explain target and preprocessing leakage.
- [ ] I can identify post-outcome features.
- [ ] I understand why split must happen before learned preprocessing.
- [ ] I can build a numeric Pipeline.
- [ ] I can build a mixed ColumnTransformer.
- [ ] I can fit and predict using the same pipeline.
- [ ] I understand pipeline use in cross-validation.
- [ ] I can audit entity, time and test-set leakage.

**Next lesson:** Day 17 — Matplotlib and Seaborn for ML Visualization.
