# Day 09: Dataset Structure, Features, Target Variables and Observation Units

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Read a dataset schema and identify its structure.
- Explain rows, columns, observations, features, targets and identifiers.
- Identify the observation unit and prediction unit.
- Distinguish numerical, categorical, ordinal, datetime and text variables.
- Define a target variable for a business prediction problem.
- Detect target leakage and ambiguity in dataset definitions.

## 2. What Is a Dataset?

A dataset is an organized collection of related observations. In a common tabular dataset, rows represent records and columns represent attributes of those records.

Example: a bank loan dataset.

| application_id | age | annual_income | loan_amount | previous_defaults | loan_default |
|---|---:|---:|---:|---:|---|
| L101 | 27 | 450000 | 150000 | 0 | No |
| L102 | 42 | 720000 | 300000 | 1 | Yes |
| L103 | 35 | 600000 | 180000 | 0 | No |

Each row represents one loan application (in this example). Each column stores one attribute.

## 3. Observation / Sample / Record

An observation is one unit represented by a row. The observation unit must be defined before analyzing or training a model.

Examples:

| Dataset | Possible observation unit |
|---|---|
| Customer churn | One customer at a defined snapshot date |
| Retail sales | One order line or one order, depending on the table |
| Medical outcomes | One patient visit or one patient |
| Fraud detection | One transaction |
| Predictive maintenance | One machine reading or one machine-time window |

**Critical question:** What does one row represent?

If this is misunderstood, counts and model splits can be wrong. For example, the same customer may appear in many monthly rows. Splitting rows randomly can put one customer's records in both training and testing sets and make performance look too optimistic for new-customer prediction.

## 4. Features and Target Variable

### Features (X)

Features are the input variables used by a model to make a prediction.

In the loan example, potential features might include annual income, loan amount and previous defaults, subject to policy and governance review.

### Target (y)

The target is the outcome the model is trained to predict.

In the example, `loan_default` is the target.

```text
Dataset
  |
  +----> Features (X): age, income, loan amount, past history
  |
  +----> Target (y): loan default (Yes / No)
```

For supervised learning, training examples need target values. An unlabeled dataset may still be useful for unsupervised learning, but not directly for ordinary supervised target fitting.

### Separate X and y in Pandas

```python
import pandas as pd

# Assume df has the columns shown in the table.
X = df[["age", "annual_income", "loan_amount", "previous_defaults"]]
y = df["loan_default"]

print(X.shape)
print(y.shape)
```

Use double brackets for a DataFrame of multiple features. A single bracket selection returns a Series for the target.

## 5. Types of Features

| Type | Meaning | Example | Typical handling |
|---|---|---|---|
| Numerical — discrete | Countable numeric values | Number of claims | Numeric validation; sometimes count-aware transforms |
| Numerical — continuous | Measured values | Temperature, income | Numeric validation, scaling where needed |
| Nominal categorical | Categories without order | City, browser type | One-hot/other suitable encoding |
| Ordinal categorical | Categories with meaningful order | Low, Medium, High | Explicit order-aware encoding if justified |
| Binary | Two categories | Active / inactive | Boolean or 0/1 encoding |
| Datetime | Date/time values | Signup date | Extract valid time features; respect time boundaries |
| Text | Free-form language | Customer complaint | Text processing/embeddings depending on task |
| Identifier | Identifies a record/entity | Customer ID | Usually exclude as a direct numeric signal unless carefully justified |

A column storing numbers is not always a numerical measurement. A customer ID like 10023 is an identifier, not an amount where 10023 is greater than 10022 in a useful predictive sense.

## 6. Feature vs Target vs Identifier

| Column | Role in churn dataset | Reason |
|---|---|---|
| Customer ID | Identifier | Identifies customer; usually not a behavioural measurement |
| Months subscribed | Feature | Historical tenure may help estimate churn risk |
| Support calls in last 30 days | Feature | Available behaviour before prediction time |
| Churned next month | Target | Defined future outcome to predict |
| Cancellation confirmation date | Potential leakage | May only exist after churn occurs |

Do not select every column except the target automatically. Understand each field, availability time and business meaning.

## 7. Define the Prediction Problem Correctly

A usable ML dataset definition should answer:

1. **Who/what is being predicted?** One customer, order, transaction or device?
2. **When is the prediction made?** At signup, at the start of each month, before payment, etc.?
3. **What period does the target cover?** Churn in the next 30 days, default within 12 months, etc.?
4. **Which information is available at prediction time?** Only past and current information available then.
5. **What does each row represent?** One record per entity or one entity-time snapshot?

Without these definitions, model results may be meaningless even if the Python code runs.

## 8. Data Schema and Data Dictionary

A schema describes the structure of the dataset: field names, data types and sometimes constraints. A data dictionary explains what each field means.

Example:

| Column | Type | Definition | Allowed values / unit |
|---|---|---|---|
| customer_id | string | Unique customer identifier | Unique non-empty ID |
| age | integer | Age at snapshot date | Years; validate plausible range |
| monthly_charge | decimal | Monthly subscription charge | Currency amount |
| plan_type | category | Subscription plan | Basic, Plus, Premium |
| churn_next_30d | category | Whether customer churned in next 30 days | Yes / No |

A clear data dictionary reduces inconsistent interpretations between analysts, engineers and business stakeholders.

## 9. Practical Schema Inspection with Pandas

```python
import pandas as pd

# Load data
# df = pd.read_csv("customers.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:")
print(df.dtypes)
print("Sample rows:")
print(df.head())
print("Unique values:")
print(df.nunique(dropna=False))
```

What each output helps answer:

- Shape: how many rows and columns were loaded?
- Columns: what fields exist, including spelling?
- Dtypes: are numeric columns actually numeric?
- Head: do sample values match expectations?
- Nunique: are columns constant, identifier-like or low-cardinality?

These checks are an initial profile, not a complete validation.

## 10. Industry Scenario — Subscription Churn Dataset

A subscription business supplies a table with 50,000 rows and 14 columns. The manager asks for a model that identifies customers likely to cancel during the next 30 days.

Before training, the ML engineer clarifies:

- Is one row one customer or one customer per month?
- What date is the prediction made?
- Does the target represent cancellation in the following 30 days?
- Are support calls counted only up to the prediction date?
- Is cancellation date included as a column? If so, it may reveal the target.
- Can one customer appear across multiple snapshots?

The engineer documents the observation unit, target window and feature availability, then prepares a leakage-safe split.

**Key lesson:** A dataset's row and column counts do not define a good ML problem. The meaning and timing of each record are essential.

## 11. Common Mistakes

- Assuming each row is a unique customer when it may be a monthly snapshot.
- Treating an ID as a continuous numerical feature.
- Choosing a target without defining its future time window.
- Including information that becomes available only after the event.
- Mixing multiple target definitions in one column.
- Ignoring units, categories and data dictionary definitions.
- Randomly splitting repeated entities without considering grouping or time.

## 12. Quick Revision

- A dataset is structured observations and attributes.
- Observation unit defines what one row means.
- Features are inputs; target is the supervised outcome.
- IDs identify entities but are not automatically useful measurements.
- Schema and data dictionary describe fields and meanings.
- Prediction time and target window must be defined to prevent leakage.

---

## Beginner-to-Developer Expansion: More Examples and Real-World Reasoning

### A. Start with the simplest meaning
Before memorizing a function, explain the idea in your own words. Identify the input, output and the reason the operation exists. A correct syntax call on the wrong data is still a wrong solution.

### B. Additional practical scenario
A churn table may have one row per customer-month. The observation unit is customer-month, not customer. If one customer appears in multiple months, split strategy must prevent identity leakage when the goal is performance on unseen customers.

### C. Syntax and use: how to choose
Features are known inputs at prediction time; target is the outcome being predicted; row grain says what one row represents; identifiers distinguish records but are not automatically predictive measurements.

### D. What happens if this is skipped?
If the row grain or prediction timestamp is wrong, duplicated entities and post-outcome fields can leak into evaluation.

### E. Senior developer review checklist
- Is the business definition clear?
- Are input types, units, shape and missing values checked?
- Can the result be verified on a small hand-checkable example?
- Have normal, boundary and invalid cases been considered?
- Is the approach simple enough to explain and maintain?
- Are limitations and possible failure costs documented?

### F. Explain it in an interview
Use this order: definition → why it is needed → syntax or method → small example → real business scenario → what can go wrong → how you validate the result. Avoid answering with only a library function name.
