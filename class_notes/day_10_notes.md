# Day 10: Missing Values, Duplicates and Inconsistent Data

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Identify missing values and understand why they occur.
- Distinguish missing values from valid zero, blank text and special codes.
- Detect exact and business-key duplicates.
- Identify inconsistent formats, labels, units and invalid values.
- Apply suitable cleaning options using Pandas.
- Explain how data-quality problems can distort ML training and evaluation.
- Document cleaning decisions so the process is reproducible.

## 2. What Is Data Cleaning?

Data cleaning is the process of identifying, investigating and appropriately correcting or handling quality problems in a dataset.

Common problems:

| Problem | Example |
|---|---|
| Missing value | Customer age is empty |
| Duplicate | Same transaction loaded twice |
| Inconsistent text | `Hyderabad`, `hyderabad`, `HYD` |
| Invalid value | Age = -5 |
| Wrong data type | Price stored as `"₹1,200"` text |
| Inconsistent unit | Weight recorded partly in kg and partly in lb |
| Invalid date | `31/02/2026` |

Cleaning should preserve valid information. It is not simply deleting unusual or inconvenient rows.

## 3. Missing Values

A missing value means the field has no recorded value. In Pandas, missing data is commonly represented by `NaN`, `None` or `NaT`, depending on the data type.

### 3.1 Why values may be missing

- Customer did not provide the information.
- Sensor or system failed to record it.
- Data transfer or extraction failed.
- Field was not applicable to that record.
- Data was intentionally withheld.

The reason matters. Missingness can itself contain useful information, but it can also reflect a process problem or introduce bias.

### 3.2 Inspect missing values

```python
import pandas as pd

# Example data

df = pd.DataFrame({
    "customer": ["A", "B", "C", "D"],
    "age": [25, None, 40, None],
    "income": [30000, 45000, None, 52000]
})

print(df.isna())
print(df.isna().sum())
print(df.isna().mean() * 100)
```

Expected counts:

- `age`: 2 missing values (50%).
- `income`: 1 missing value (25%).

`isna()` creates a Boolean mask, `sum()` counts `True` values by column, and `mean() * 100` calculates the missing percentage.

### 3.3 Missing-value handling options

| Method | When it may be suitable | Risk / caution |
|---|---|---|
| Drop rows | Very few rows missing and omission is safe | Can bias data if missingness is systematic |
| Drop column | Column has little value or excessive missingness | May remove an important predictor |
| Mean imputation | Numeric data with suitable distribution and limited impact | Sensitive to outliers; reduces variance |
| Median imputation | Numeric data that is skewed or has outliers | Hides uncertainty and can weaken relationships |
| Mode imputation | Categorical column with a meaningful common category | Can overrepresent the mode |
| Constant / explicit category | Missing itself has a meaningful interpretation | Must distinguish unknown from not applicable |
| Model-based imputation | Complex relationships justify added complexity | More assumptions and operational complexity |

### 3.4 Examples of basic imputation

```python
# Fill a numeric column with its median
median_age = df["age"].median()
df["age"] = df["age"].fillna(median_age)

# Fill a categorical column with an explicit marker
# df["city"] = df["city"].fillna("Unknown")
```

For ML, calculate imputation values using training data only. Fit the imputer on the training set and apply the learned transformation to validation, test and future data. Fitting an imputer on the full dataset leaks information from held-out data into training.

### 3.5 Do not confuse missing with zero

- `0` may mean a real numeric value.
- `NaN` means no value is recorded.
- `-1` may be a special code but must be confirmed in the data dictionary.
- `"Unknown"` is text and may need standardized treatment.

Replacing all missing ages with zero would imply customers have age zero, which is not a sensible default for an ordinary adult customer dataset.

## 4. Duplicate Records

A duplicate occurs when a record is repeated according to a chosen definition.

### 4.1 Exact duplicates

```python
print(df.duplicated().sum())
```

`duplicated()` marks repeated rows after the first occurrence by default.

To view them:

```python
duplicates = df[df.duplicated(keep=False)]
print(duplicates)
```

`keep=False` marks every member of a duplicated group.

### 4.2 Remove exact duplicates

```python
df = df.drop_duplicates()
```

This removes exact duplicate rows, keeping the first occurrence by default. Before applying it, verify that repeated rows are truly duplicate records rather than repeated legitimate events.

### 4.3 Business-key duplicates

A business key defines which fields should uniquely identify a record.

Example: order lines might be unique by `order_id` and `line_id`, not by customer and amount.

```python
print(df.duplicated(subset=["order_id", "line_id"], keep=False))
```

Do not delete records based on an arbitrary subset. Two genuine purchases can have the same customer and amount.

## 5. Inconsistent Data

### 5.1 Inconsistent text labels

Examples:

- `Hyderabad`, `hyderabad`, `HYDERABAD`
- `Premium`, `premium `, `PREMIUM`

Standardize only when they refer to the same category.

```python
df["city"] = df["city"].str.strip().str.title()
```

This removes leading/trailing spaces and standardizes title case. It does not automatically resolve abbreviations or different spellings such as `Hyd` and `Hyderabad`; use a documented mapping after confirming equivalence.

### 5.2 Map known category variants

```python
city_mapping = {
    "Hyd": "Hyderabad",
    "HYD": "Hyderabad",
    "Bangalore": "Bengaluru"
}

df["city"] = df["city"].replace(city_mapping)
```

Use mappings based on a verified business glossary, not guesswork.

### 5.3 Numeric values stored as text

```python
df["amount"] = (
    df["amount"].astype("string")
      .str.replace("₹", "", regex=False)
      .str.replace(",", "", regex=False)
      .str.strip()
)

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
```

This example removes a rupee symbol and thousands separators before conversion. Invalid strings become missing values; count them and investigate the raw source.

### 5.4 Validate ranges and categories

```python
invalid_age = df[(df["age"] < 0) | (df["age"] > 110)]
print(invalid_age)
```

A range check flags suspicious values, but domain rules must determine acceptable ranges. Do not silently clamp or remove values without investigation.

## 6. A Practical Data-Quality Inspection Sequence

```text
Load source data
      |
      v
Inspect schema and row counts
      |
      v
Check missing values
      |
      v
Check exact and key-based duplicates
      |
      v
Check types, formats, units and categories
      |
      v
Validate ranges and business rules
      |
      v
Apply documented corrections/handling
      |
      v
Re-check counts and quality rules
      |
      v
Save clean output and cleaning log
```

## 7. End-to-End Small Cleaning Example

```python
import pandas as pd

raw = pd.DataFrame({
    "order_id": ["O1", "O2", "O2", "O3", "O4"],
    "city": ["hyderabad", "Hyd ", "Hyd ", "CHENNAI", None],
    "amount": ["₹1,200", "800", "800", "bad-value", "500"],
    "quantity": [2, 1, 1, 0, 3]
})

# 1. Remove exact duplicate rows
clean = raw.drop_duplicates().copy()

# 2. Standardize text formatting
clean["city"] = clean["city"].str.strip().str.title()
clean["city"] = clean["city"].replace({"Hyd": "Hyderabad"})

# 3. Convert amount to a numeric column
clean["amount"] = (
    clean["amount"].astype("string")
         .str.replace("₹", "", regex=False)
         .str.replace(",", "", regex=False)
)
clean["amount"] = pd.to_numeric(clean["amount"], errors="coerce")

# 4. Flag invalid quantity (zero is not valid for this example's order line)
clean["invalid_quantity"] = clean["quantity"] <= 0

print(clean)
print("Missing by column:")
print(clean.isna().sum())
```

Expected observations:

- The repeated O2 row is removed as an exact duplicate.
- `Hyd` becomes `Hyderabad` using the explicitly supplied mapping.
- `bad-value` becomes missing after numeric conversion and should be investigated.
- A missing city remains missing; the code does not invent a city.
- Quantity 0 is flagged for review rather than automatically deleted.

## 8. Why Data Cleaning Matters for ML

| Quality issue | Possible ML impact |
|---|---|
| Missing values | Estimator may fail or learn biased patterns |
| Duplicate rows | Repeated records can overweight examples and distort evaluation |
| Inconsistent categories | Same business category may be treated as multiple categories |
| Wrong units | Feature magnitudes become incorrect |
| Invalid values | Model learns from impossible or erroneous examples |
| Future information | Data leakage creates unrealistic evaluation |

Cleaning must be learned from training data where applicable and consistently applied to future records. A reproducible preprocessing pipeline is safer than manually cleaning training and production data in different ways.

## 9. Industry Scenario — Retail Sales Dataset

A retailer imports data from three regional billing systems. The merged dataset contains inconsistent city names, amounts with currency symbols, missing product IDs and repeated order rows.

The ML engineer should:

1. Confirm the row definition (order or order line).
2. Establish the correct unique key.
3. Compare source row counts before and after merging.
4. Standardize region names using a verified mapping.
5. Parse currency values and record invalid conversions.
6. Investigate missing product IDs with the source system.
7. Distinguish duplicated imports from legitimate repeat purchases.
8. Re-run quality checks and preserve a cleaning log.

**Interview insight:** Do not say "I removed all nulls and duplicates" as a universal solution. Explain how you identified the issue, validated its cause and selected a handling strategy.

## 10. Common Mistakes

- Dropping every row with one missing field without checking impact.
- Replacing missing values with zero without semantic justification.
- Removing duplicates based only on customer ID when multiple orders are valid.
- Standardizing category names using unverified assumptions.
- Converting invalid numeric strings to missing values without counting them.
- Fitting imputation statistics on the complete dataset before splitting.
- Failing to compare row counts and quality metrics after cleaning.

## 11. Quick Revision

- Identify why data is missing before choosing a treatment.
- Exact duplicate and business-key duplicate are different concepts.
- Standardize categories only with a verified mapping.
- Validate data types, units, ranges and business constraints.
- Fit learned preprocessing steps using training data only.
- Document every material cleaning decision and verify the output.