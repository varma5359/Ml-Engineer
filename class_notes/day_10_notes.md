# Day 10: Missing Values, Duplicates and Inconsistent Data

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define missing values and identify why they occur.
- Distinguish missing, blank, zero and special-code values.
- Detect exact duplicates and business-key duplicates.
- Identify inconsistent text, dates, units and categories.
- Apply basic Pandas cleaning operations.
- Explain why cleaning decisions must preserve business meaning.

## 1. What Is Data Cleaning?
Data cleaning is the process of identifying and handling incorrect, incomplete, duplicated or inconsistent data so that analysis and modelling use trustworthy information.

Poor-quality data can create incorrect summaries, misleading model patterns and unreliable business decisions.

## 2. Missing Values
**Definition:** A missing value means the information was not recorded or is unavailable in that field.

Possible causes:
- User did not provide the value.
- Source system failed to capture it.
- Data integration did not find a matching record.
- The field was not applicable.

Missing is not the same as zero. A missing salary does not mean salary is ₹0.

### Detect missing values
```python
df.isna().sum()
df.isna().mean() * 100
```

**Example 1 — Count missing values**
```python
df = pd.DataFrame({"age": [25, None, 30]})
print(df["age"].isna().sum())
```
Output: 1.

**Example 2 — Remove rows missing a required target**
```python
df_clean = df.dropna(subset=["target"])
```
Use only when rows without the target cannot be used for supervised training and removal is justified.

**Example 3 — Fill missing numeric values**
```python
df["age"] = df["age"].fillna(df["age"].median())
```
Median imputation is one option; select a strategy based on the feature distribution and why values are missing.

## 3. Duplicates
**Definition:** Duplicate records repeat the same record or business event. Exact duplicate rows and duplicate business keys are not always the same.

**Example 1 — Count exact duplicate rows**
```python
print(df.duplicated().sum())
```

**Example 2 — Remove exact duplicates**
```python
df = df.drop_duplicates()
```

**Example 3 — Check a business key**
```python
print(df["transaction_id"].duplicated().sum())
df = df.drop_duplicates(subset=["transaction_id"], keep="last")
```
Only use transaction_id as the key if the business confirms that one transaction ID should identify one unique transaction. Do not remove legitimate line items merely because they share an order ID.

## 4. Inconsistent Data
**Definition:** Inconsistent data represents the same meaning in different formats or contains values that violate agreed business rules.

Examples:
- "Hyderabad", "hyderabad" and " HYDERABAD " represent the same city.
- Dates use different formats.
- A price column mixes rupees and dollars.
- A quantity contains negative values when negative quantities are not allowed by the business process.

**Example 1 — Standardize text**
```python
df["city"] = df["city"].str.strip().str.lower()
```

**Example 2 — Convert dates**
```python
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
```
Invalid date strings become NaT with errors="coerce"; inspect how many were converted.

**Example 3 — Validate numeric range**
```python
invalid_age = df[(df["age"] < 0) | (df["age"] > 120)]
print(invalid_age)
```
The allowed range must be confirmed for the application; do not treat an arbitrary threshold as universal.

## 5. A Safe Cleaning Sequence
1. Keep an untouched copy of the source data.
2. Inspect row grain, schema and data types.
3. Count missing values and duplicates.
4. Check categories, units, dates and domain constraints.
5. Decide treatment using business meaning.
6. Apply cleaning consistently.
7. Compare row counts and summary statistics before and after.
8. Document each rule and its impact.

## 6. Scenario Questions
1. An age field is missing. Should you replace it with zero? No; zero has a different meaning. Investigate and choose a suitable treatment.
2. An order ID appears five times. Does that prove duplicate transactions? No; an order may contain multiple line items. Check the row grain and unique business key.
3. Revenue suddenly doubles after loading a file. What should you inspect? Duplicate loads, join multiplication, currency units and aggregation logic.

## 7. Common Mistakes
- Filling every missing value with zero.
- Removing duplicates without defining the correct key.
- Standardizing categories without preserving the original source.
- Converting invalid values without measuring how many were affected.
- Cleaning before understanding what each row represents.
- Failing to document changes and their effect on the dataset.
