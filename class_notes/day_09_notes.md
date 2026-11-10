# Day 09: Dataset Structure, Features, Target Variables and Observation Units

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain rows, columns, observations and dataset schema.
- Identify the observation unit (row grain).
- Distinguish features, target and identifiers.
- Recognize numeric, categorical, ordinal, datetime and text columns.
- Define the prediction target and prediction time.
- Identify possible leakage and ambiguous dataset structure.

## 1. What Is a Dataset?
A dataset is an organized collection of records. In a tabular dataset, rows represent observations and columns represent attributes.

Example: a customer table may contain customer_id, age, city, monthly_spend and churned.

## 2. What Is a Schema?
A schema describes the structure of a dataset: column names, data types, constraints and sometimes relationships between tables.

**Pandas inspection syntax**
```python
import pandas as pd
df = pd.read_csv("customers.csv")
print(df.shape)
print(df.columns)
print(df.dtypes)
df.info()
```

**Three ways to inspect structure**
1. `df.shape` returns row and column counts.
2. `df.head()` previews records.
3. `df.info()` shows data types and non-null counts.

## 3. Observation and Observation Unit
An **observation** is one recorded example. The **observation unit** explains what one row represents.

Examples:
1. One row per customer.
2. One row per customer per month.
3. One row per individual transaction.

The row grain matters. A customer-month table may contain many rows for one customer; it is not one row per customer.

## 4. Features, Target and Identifier
- Feature: input variable supplied to the model.
- Target: outcome a supervised model is trained to predict.
- Identifier: value used to identify a record or entity; it is not automatically a useful predictive feature.

**Example:** In a churn dataset, age, plan_type and monthly_spend may be features; churned is the target; customer_id is an identifier.

### Three target examples
1. Predict house price → target is sale_price.
2. Predict whether a customer churns → target is churned.
3. Predict delivery duration → target is delivery_minutes.

## 5. Common Feature Types
| Type | Example | Notes |
|---|---|---|
| Numerical | age, salary | May be discrete or continuous |
| Categorical nominal | city, payment_method | No natural order |
| Categorical ordinal | low, medium, high | Has a meaningful order |
| Datetime | order_date | Often derive time features carefully |
| Text | customer_review | May need text processing |

## 6. Prediction Time and Leakage
A feature must be available at the time the prediction is made. A field recorded after the target event can leak the answer.

**Example:** Predicting whether a customer will cancel next month using a cancellation_reason field recorded after cancellation is leakage.

## 7. Simple Schema Example
```python
df = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "age": [25, 31, 28],
    "monthly_spend": [1200, 1800, 900],
    "churned": [0, 1, 0]
})
print(df.shape)
print(df.dtypes)
```
Shape is (3, 4). There are three observations and four columns.

## 8. Scenario Questions
1. A table has 10,000 rows but only 2,000 unique customer IDs. What should you investigate? Whether rows represent customer-months, transactions or duplicates.
2. You are predicting churn next month. Should a field recorded after cancellation be a feature? No.
3. Is customer_id always a useful numeric feature? No; it is an identifier and can encourage memorization.
