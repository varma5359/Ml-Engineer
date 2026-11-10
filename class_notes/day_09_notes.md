# Day 09: Dataset Structure, Features, Target Variables and Observation Units

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain what a dataset, record, observation and schema mean.
- Identify the observation unit or row grain.
- Distinguish feature columns, target columns and identifiers.
- Recognize numeric, categorical, ordinal, datetime and text data.
- Understand why a column's data type is not the same as its business meaning.
- Define the prediction target and the time when a prediction is made.
- Inspect a dataset systematically using Pandas.
- Identify common schema problems before cleaning or model training.

## 2. Why should an ML Engineer understand dataset structure?

Before training a model, an engineer must understand what one row represents, what each column means and which information is available when a prediction is made.

Imagine an online store wants to predict whether a customer will cancel a subscription. The data team provides a CSV file with customer details, payments, service history and cancellation status.

If the engineer assumes every row represents one customer but the file actually contains one row per customer per month, the analysis and train-test split may be wrong. A model may appear to perform well because it sees records from the same customer in both training and testing.

**Main principle:** Understand the observation unit and the meaning of each column before choosing preprocessing steps or an ML algorithm.

## 3. What is a dataset?

A **dataset** is an organized collection of records about a particular subject or process.

In a tabular dataset:
- Rows usually represent observations or records.
- Columns represent attributes or fields.
- A cell contains the value for one field in one record.

### 3.1 A small customer dataset

| customer_id | age | city | monthly_spend | churn |
|---|---:|---|---:|---:|
| C001 | 25 | Hyderabad | 4500 | 0 |
| C002 | 34 | Chennai | 5200 | 1 |
| C003 | 29 | Vizag | 6100 | 0 |

In this example, each row represents one customer at a particular snapshot. The exact snapshot date should be defined in a real project.

### 3.2 Create a small dataset in Pandas

### 3.1. Import Pandas

```python
import pandas as pd

print(pd.__version__)
```

**Expected output**

```text
A Pandas version string
```

**Explanation**

The `pd` alias is the common convention. Version information helps reproduce the environment.



### 3.2. Create a customer table

```python
data = {
    "customer_id": ["C001", "C002", "C003"],
    "age": [25, 34, 29],
    "city": ["Hyderabad", "Chennai", "Vizag"],
    "monthly_spend": [4500, 5200, 6100],
    "churn": [0, 1, 0]
}

df = pd.DataFrame(data)
print(df)
```

**Expected output**

```text
A table with three rows and five columns
```

**Explanation**

Dictionary keys become column names. Each list provides values for that column. Each row represents one customer in this teaching example.



## 4. Important dataset vocabulary

| Term | Meaning | Example |
|---|---|---|
| Dataset | Complete collection being analysed | Customer churn table |
| Record | One stored row | Customer C001 |
| Observation | One example measured for analysis | Customer at a snapshot date |
| Feature | Input information used for prediction | Age or monthly spend |
| Target | Outcome to be predicted | Churn |
| Identifier | Field used to distinguish records/entities | Customer ID |
| Schema | Definition of columns, types and constraints | Age is numeric and non-negative |
| Row grain | What one row represents | One customer per month |
| Snapshot | The point in time represented by a record | Customer status on 31 October |

### 4.1 Why vocabulary matters

The phrase "customer dataset" is not precise enough. It may contain:
- One row per customer.
- One row per customer per month.
- One row per transaction.
- One row per customer interaction.
- One row per subscription.

These structures support different questions and require different validation strategies.

## 5. What is a schema?

A **schema** describes the expected structure of a dataset. It may specify:
- Column names.
- Data types.
- Allowed values.
- Required or optional fields.
- Uniqueness constraints.
- Relationships between tables.
- Units and definitions.
- The observation unit and time meaning.

A CSV can load successfully even when its schema is wrong. For example, an age column may contain text, a customer ID may lose leading zeros, or a target column may contain unexpected labels.

### 5.1 Inspect the first few rows

### 5.1. Preview records using head()

```python
print(df.head())
```

**Expected output**

```text
The first five rows, or all rows if fewer than five exist
```

**Explanation**

This is a quick visual check. It does not validate the complete file.



### 5.2 Inspect the shape

### 5.2. Read row and column counts

```python
print(df.shape)
```

**Expected output**

```text
(3, 5)
```

**Explanation**

The first value is the number of rows and the second is the number of columns.



### 5.3 Inspect the column names

### 5.3. Read the column labels

```python
print(df.columns.tolist())
```

**Expected output**

```text
['customer_id', 'age', 'city', 'monthly_spend', 'churn']
```

**Explanation**

The column list helps confirm that required fields are present and named as expected.



### 5.4 Inspect data types

### 5.4. Inspect dtypes

```python
print(df.dtypes)
```

**Expected output**

```text
customer_id object, age int64, city object, monthly_spend int64, churn int64 (typical display)
```

**Explanation**

Pandas assigns a dtype based on the loaded values. A dtype is a technical representation, not a complete description of business meaning.



### 5.5 Inspect non-null counts

### 5.5. Use info()

```python
df.info()
```

**Expected output**

```text
A summary with row count, column names, non-null counts, dtypes and memory usage
```

**Explanation**

This helps identify columns that contain missing values and columns whose types may need investigation.



### 5.6 Inspect summary statistics

### 5.6. Use describe()

```python
print(df.describe())
```

**Expected output**

```text
Summary statistics for numeric columns
```

**Explanation**

The summary includes count, mean, standard deviation, minimum, quartiles and maximum. It does not prove values are valid.



## 6. Observation unit — what does one row represent?

The **observation unit** or **row grain** explains what a single row represents.

This is one of the most important questions in a machine-learning project.

### 6.1 One row per customer

| customer_id | age | total_spend | churn |
|---|---:|---:|---:|
| C001 | 25 | 12000 | 0 |
| C002 | 34 | 9000 | 1 |

Each customer appears once in this table.

### 6.2 One row per customer per month

| customer_id | month | monthly_spend | churn_next_month |
|---|---|---:|---:|
| C001 | Jan | 1000 | 0 |
| C001 | Feb | 1200 | 0 |
| C002 | Jan | 800 | 1 |

A customer can appear many times. This is a customer-month dataset, not a one-row-per-customer dataset.

### 6.3 One row per transaction

| transaction_id | customer_id | amount | payment_status |
|---|---|---:|---|
| T001 | C001 | 500 | success |
| T002 | C001 | 250 | failed |
| T003 | C002 | 900 | success |

A customer can have many transactions. Treating each transaction as a separate customer would be incorrect.

### 6.4 Questions to establish row grain

Before modelling, ask:
1. What does one row represent?
2. Can the same customer appear in multiple rows?
3. Is there a unique row identifier?
4. Is the timestamp a creation time, event time or snapshot time?
5. Are multiple rows for one entity expected?
6. What is the prediction task's unit: customer, transaction, product or time period?

## 7. Check identifiers and uniqueness

An identifier is a field used to identify a record or entity. It is not automatically a useful predictive feature.

### 7.1 Check whether customer IDs are unique

### 7.1. Count unique customer IDs

```python
print(df["customer_id"].nunique())
```

**Expected output**

```text
3
```

**Explanation**

`nunique()` counts distinct non-missing values. Compare it with the row count to investigate possible repeated entities.



### 7.2 Check duplicate customer IDs

### 7.2. Count repeated IDs

```python
print(df["customer_id"].duplicated().sum())
```

**Expected output**

```text
0
```

**Explanation**

This counts ID values repeated after their first occurrence. If the intended row grain is one row per customer, repeated IDs need investigation.



### 7.3 Check whether all rows have unique IDs

### 7.3. Compare row count and unique ID count

```python
print(len(df))
print(df["customer_id"].nunique())
```

**Expected output**

```text
3 followed by 3
```

**Explanation**

Matching counts suggest unique non-missing IDs in this small example. It does not validate that IDs are correctly assigned or stable across systems.



### 7.4 Check missing identifiers

### 7.4. Count missing customer IDs

```python
print(df["customer_id"].isna().sum())
```

**Expected output**

```text
0
```

**Explanation**

A missing identifier can prevent reliable record matching. Define how such records should be investigated.



### 7.5 Do not use an ID as a numeric measurement

A customer ID such as 1005 is usually a label, not a quantity. The fact that 1005 is greater than 1004 does not mean that customer 1005 has more value or risk.

**ML decision:** Keep identifiers available for joining, grouping and auditing where needed, but do not automatically include them as model features.

## 8. Feature, target and identifier

### Feature

A **feature** is an input variable used by a model to make a prediction. Examples include age, monthly spend, tenure and previous support calls.

### Target

A **target** is the outcome a supervised-learning model is trained to predict. Examples include churn, house price, delivery delay or fraud label.

### Identifier

An **identifier** distinguishes a record or entity. Examples include customer ID, transaction ID and account number.

### 8.1 Separate feature columns and target

### 8.1. Select the target column

```python
y = df["churn"]

print(y)
```

**Expected output**

```text
A Series containing 0, 1 and 0
```

**Explanation**

The target column contains the outcome for each customer. In a real problem, confirm the target definition and when it was recorded.



### 8.2. Select feature columns

```python
feature_columns = ["age", "city", "monthly_spend"]
X = df[feature_columns]

print(X)
```

**Expected output**

```text
A DataFrame with age, city and monthly_spend
```

**Explanation**

The feature table contains input information. The categorical city field will need suitable encoding before many estimators can use it.



### 8.2 Verify feature and target row counts

### 8.3. Compare the number of observations

```python
print(X.shape[0])
print(y.shape[0])
print(X.shape[0] == y.shape[0])
```

**Expected output**

```text
3, 3, True
```

**Explanation**

The row counts match. In addition, verify that X and y refer to the same observations in the same order.



### 8.3 Exclude the identifier from the initial feature set

### 8.4. Inspect the selected feature columns

```python
print(X.columns.tolist())
```

**Expected output**

```text
['age', 'city', 'monthly_spend']
```

**Explanation**

The ID is intentionally not included in this initial feature list. This is a starting decision, not a universal rule; some identifiers may contain legitimate time or group information that should be represented carefully.



## 9. Recognize common data types

A column's type helps determine which operations are appropriate. The data type must be interpreted with the column's meaning.

| Type | Meaning | Examples | Common preparation |
|---|---|---|---|
| Integer | Whole-number representation | Age, count | Validate range and meaning |
| Float | Decimal numeric representation | Price, temperature | Check units and invalid values |
| Nominal category | Labels without natural order | City, payment type | One-hot or suitable categorical encoding |
| Ordinal category | Categories with meaningful order | Low, medium, high | Order-aware encoding |
| Boolean | True/False flag | Is_active | Convert consistently |
| Datetime | Date or time | Signup date | Parse, derive time features carefully |
| Text | Free-form words | Review, complaint | Text processing if required |
| Identifier | Record/entity key | Customer ID | Preserve for joins/audits; assess leakage |

### 9.1 Inspect a mixed-type dataset

### 9.1. Create mixed data types

```python
mixed = pd.DataFrame({
    "age": [25, 34, 29],
    "spend": [4500.5, 5200.0, 6100.75],
    "city": ["Hyderabad", "Chennai", "Vizag"],
    "is_active": [True, False, True]
})

print(mixed.dtypes)
```

**Expected output**

```text
age int64, spend float64, city object, is_active bool (typical)
```

**Explanation**

Pandas detects different types for different columns. The exact string dtype representation can vary with how the data was created or read.



### 9.2 Inspect unique category values

### 9.2. List unique cities

```python
print(df["city"].unique())
```

**Expected output**

```text
An array containing Hyderabad, Chennai and Vizag
```

**Explanation**

Unique values help reveal spelling differences, inconsistent capitalization and unexpected categories.



### 9.3 Count categories

### 9.3. Count city frequencies

```python
print(df["city"].value_counts())
```

**Expected output**

```text
One count for each city
```

**Explanation**

Frequency counts show how many rows belong to each category. For repeated entities, this is a row count, not necessarily a unique-customer count.



### 9.4 Inspect numeric range

### 9.4. Check minimum and maximum age

```python
print(df["age"].min())
print(df["age"].max())
```

**Expected output**

```text
25 followed by 34
```

**Explanation**

Range checks help identify impossible or unusual values. A plausible range must come from the domain, not guesswork.



## 10. Define the target carefully

A target definition must specify:
- The event or quantity being predicted.
- The entity or observation unit.
- The time horizon.
- The label creation rule.
- How unknown or incomplete outcomes are handled.

For churn prediction, "churn" could mean:
- Subscription cancelled today.
- No purchase for 90 days.
- Account inactive for a defined period.
- Customer did not renew at contract expiry.

These are not interchangeable definitions.

### 10.1 Inspect target labels

### 10.1. Count target classes

```python
print(df["churn"].value_counts(dropna=False))
```

**Expected output**

```text
Counts for target values 0 and 1
```

**Explanation**

Check whether labels are balanced and whether unexpected or missing target values exist.



### 10.2 Inspect target proportions

### 10.2. Calculate target proportions

```python
print(df["churn"].value_counts(normalize=True, dropna=False))
```

**Expected output**

```text
Proportions for each target value
```

**Explanation**

Normalized counts show the share of rows in each class. This is a first inspection, not a full evaluation of model performance.



### 10.3 Check allowed target values

### 10.3. Inspect distinct target labels

```python
print(sorted(df["churn"].dropna().unique()))
```

**Expected output**

```text
[0, 1]
```

**Explanation**

This example expects binary labels. If additional values appear, confirm their meaning rather than silently recoding them.



## 11. Prediction time and feature availability

A feature must be available at the time when the model is expected to make its prediction.

Example:
- Prediction time: before a customer renews a subscription.
- Available information: tenure, usage history up to today, previous support calls.
- Unavailable future information: cancellation reason recorded after cancellation.

A field can be technically present in the dataset but still be invalid for prediction because it is recorded later.

### 11.1 Ask a prediction-time question for each feature

| Field | Available before prediction? | Initial decision |
|---|---|---|
| Customer tenure | Usually yes, if calculated as of prediction date | Candidate feature |
| Last month's usage | Yes, if available at prediction date | Candidate feature |
| Cancellation reason | Usually after cancellation | Exclude; leakage risk |
| Account ID | Available, but identifier | Do not automatically use |
| Future renewal status | After prediction horizon | Target or leakage risk |

## 12. Common schema problems

- Required column missing.
- Column name differs from the expected name.
- Numeric values stored as text.
- ID column parsed as integer and loses leading zeros.
- Unexpected category labels.
- Duplicate records.
- Multiple row grains mixed in one file.
- Target values outside the approved label set.
- Datetime values stored as unparsed text.
- Features contain information recorded after the prediction event.

### 12.1 Check required columns

### 12.1. Validate required column names

```python
required_columns = ["customer_id", "age", "city", "monthly_spend", "churn"]

missing_columns = [column for column in required_columns if column not in df.columns]

print(missing_columns)
```

**Expected output**

```text
[]
```

**Explanation**

An empty list means every required column name was found. This is a simple schema check.



### 12.2 Report unexpected columns

### 12.2. Find columns not in the expected schema

```python
expected_columns = set(required_columns)
actual_columns = set(df.columns)

unexpected_columns = actual_columns - expected_columns

print(unexpected_columns)
```

**Expected output**

```text
set()
```

**Explanation**

Unexpected columns are not automatically errors. Review them and decide whether they belong in the approved schema.



### 12.3 Check numeric column types

### 12.3. Inspect age and monthly spend types

```python
print(df[["age", "monthly_spend"]].dtypes)
```

**Expected output**

```text
Numeric dtypes for both columns in this example
```

**Explanation**

If a required numeric feature is stored as text, inspect the source values and conversion rules before using it.



## 13. Mini workflow: profile a dataset before modelling

Use this ordered checklist whenever a new tabular dataset arrives:

1. Confirm the file version and source.
2. Load the dataset.
3. Check shape and column names.
4. Read the schema and data types.
5. Confirm the observation unit.
6. Identify identifiers and duplicate keys.
7. Identify candidate features.
8. Define the target and prediction horizon.
9. Check target labels.
10. Verify feature availability at prediction time.
11. Record questions and data-quality issues.
12. Only then proceed to cleaning and EDA.

## 14. Interview-ready explanations

**What is an observation unit?**  
It defines what one row represents, such as one customer, one transaction or one customer-month. Understanding row grain prevents incorrect aggregation and data splitting.

**What is the difference between a feature and a target?**  
Features are inputs used to make a prediction. The target is the outcome the supervised model is trained to predict.

**Why should an ID not automatically be used as a feature?**  
An ID is usually a label, not a measurement. It may encode ordering or source-system information that does not generalize and can create leakage or spurious patterns.

**Why must features be available at prediction time?**  
The deployed model cannot use future information. A feature recorded after the outcome creates target leakage and unrealistic evaluation.

## 15. Scenario questions

1. A dataset has 50,000 rows but only 8,000 unique customer IDs. What might this mean?  
   **Answer:** It may contain multiple observations per customer. Confirm whether the row grain is transaction, month or another event.

2. A target column contains 0, 1 and "pending". What do you do?  
   **Answer:** Confirm what pending means, whether it is an incomplete outcome or a valid class, and define the label policy with the domain owner.

3. A model predicts churn but one feature is cancellation_date. What is the concern?  
   **Answer:** The date may only exist after cancellation, so it may leak the outcome. Verify when the field becomes available.

4. The customer ID column contains 0012 and 0013 but is loaded as integers. What is the risk?  
   **Answer:** Leading zeros may be lost. Read identifiers as strings when their exact representation matters.

5. The dataset shape is (10,000, 15). Does this prove there are 10,000 customers?  
   **Answer:** No. It only says there are 10,000 rows and 15 columns. The row grain and unique entity count must be checked.

## 16. Lesson completion checklist

- [ ] I can explain dataset, schema, observation and row grain.
- [ ] I can identify features, target and identifiers.
- [ ] I can inspect shape, columns, dtypes and non-null counts.
- [ ] I can check unique IDs and repeated entities.
- [ ] I can recognize common data types.
- [ ] I can inspect target labels and proportions.
- [ ] I can explain why prediction time matters.
- [ ] I can validate required columns.
- [ ] I can describe the initial dataset profiling workflow.

**Next lesson:** Day 10 — Missing Values, Duplicates and Inconsistent Data.
