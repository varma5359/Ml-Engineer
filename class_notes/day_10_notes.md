# Day 10: Missing Values, Duplicates and Inconsistent Data

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain data cleaning and why it is required before ML.
- Distinguish missing values, blank strings, zero and special codes.
- Detect missing values by row and column.
- Identify exact duplicate rows and repeated business keys.
- Standardize inconsistent text, categories and dates.
- Convert data types safely.
- Choose when to remove, replace or preserve records.
- Validate the dataset after every cleaning operation.
- Record the number of rows and values affected by cleaning.

## 2. Why is data cleaning important?

A machine-learning model learns from the data supplied to it. If records contain missing, duplicated, inconsistent or invalid values, the model may learn misleading patterns or fail during preprocessing.

Example issues in a customer dataset:
- Age is blank.
- Monthly spend is stored as text.
- City is written as "Hyderabad", "hyderabad" and "HYD".
- The same transaction is recorded twice.
- Missing salary is represented as zero.
- A date column contains multiple formats.

**Important:** Cleaning does not mean deleting every unusual value. Every action must be justified by the meaning of the data and the prediction task.

## 3. Create a small practice dataset

We will create an intentionally imperfect dataset. Each cell is a separate step so you can run it, inspect the output and understand the change before continuing.

### 3.1 Import required libraries

### 3.1. Import Pandas and NumPy

```python
import pandas as pd
import numpy as np

print("Libraries imported")
```

**Expected output**

```text
Libraries imported
```

**Explanation**

Pandas handles the table; NumPy provides values such as `np.nan` to represent missing numerical entries.



### 3.2 Create raw customer records

### 3.2. Create an imperfect dataset

```python
raw_data = {
    "customer_id": ["C001", "C002", "C003", "C003", "C005", "C006", "C007"],
    "age": [25, np.nan, 35, 35, 42, 29, 150],
    "city": ["Hyderabad", "hyderabad", "Chennai", "Chennai", "Vizag ", "VIZAG", None],
    "monthly_spend": [4500, 5200, "6100", "6100", "unknown", 3900, 7200],
    "signup_date": ["2026-01-10", "2026/01/12", "12-01-2026", "12-01-2026", "2026-02-01", None, "2026-02-10"]
}

df = pd.DataFrame(raw_data)
print(df)
```

**Expected output**

```text
A seven-row DataFrame with missing values, inconsistent categories, text in a numeric field, repeated records and a suspicious age
```

**Explanation**

This dataset is deliberately imperfect. Do not clean it all at once. First inspect it and decide which issues are real problems.



### 3.3 Preserve the raw input

### 3.3. Create a working copy

```python
raw_df = df.copy()
clean_df = df.copy()

print(raw_df.shape)
print(clean_df.shape)
```

**Expected output**

```text
(7, 5) followed by (7, 5)
```

**Explanation**

Keep the original data unchanged. Use a working copy for cleaning so you can compare results and recover the original values.



### 3.4 Inspect the first rows

### 3.4. Preview the raw data

```python
print(clean_df.head())
```

**Expected output**

```text
The first five imperfect records
```

**Explanation**

The preview gives examples, but quality checks must cover the entire dataset.



### 3.5 Inspect data types

### 3.5. Inspect dtypes

```python
print(clean_df.dtypes)
```

**Expected output**

```text
age numeric, city object, monthly_spend object, signup_date object (typical)
```

**Explanation**

The monthly_spend column is object because it mixes numeric-looking text with the word unknown. The date values are still text.



### 3.6 Inspect dimensions

### 3.6. Check the initial shape

```python
print(clean_df.shape)
```

**Expected output**

```text
(7, 5)
```

**Explanation**

Record the original row and column counts. Compare them after each major transformation.



## 4. Missing values

A **missing value** means information is not present in a field. It may be represented by `NaN`, `None`, `NaT` or another source-specific marker.

Missing values may occur because:
- A customer did not provide the information.
- A system failed to capture a value.
- A source system did not contain that field.
- A join did not find a matching record.
- A value is not applicable.

**Missing is not the same as zero.** A missing monthly spend does not mean the customer spent nothing.

### 4.1 Count missing values in each column

### 4.1. Use isna().sum()

```python
print(clean_df.isna().sum())
```

**Expected output**

```text
A missing-value count for every column
```

**Explanation**

`isna()` returns True where values are missing. `sum()` counts the True values column by column.



### 4.2 Calculate missing-value percentages

### 4.2. Calculate percentage missing

```python
missing_percent = clean_df.isna().mean() * 100

print(missing_percent.round(2))
```

**Expected output**

```text
A percentage for each column
```

**Explanation**

A Boolean True is treated as one when calculating the mean. Multiplying by 100 gives the percentage of rows missing in each column.



### 4.3 Check whether the dataset has any missing values

### 4.3. Use any()

```python
print(clean_df.isna().values.any())
```

**Expected output**

```text
True
```

**Explanation**

This is an overall check. It does not tell you where the missing values are.



### 4.4 Find rows containing at least one missing value

### 4.4. Filter rows with missing data

```python
rows_with_missing = clean_df[clean_df.isna().any(axis=1)]

print(rows_with_missing)
```

**Expected output**

```text
Rows that contain one or more missing cells
```

**Explanation**

The axis argument selects a row-wise check. This helps inspect the affected records before deciding what to do.



### 4.5 Count missing values per row

### 4.5. Count missing fields in each record

```python
print(clean_df.isna().sum(axis=1))
```

**Expected output**

```text
One missing-field count per row
```

**Explanation**

This helps identify records with many incomplete fields. A row with one missing value may be treated differently from a row missing most of its fields.



## 5. Blank strings and special missing markers

A common issue is that blank strings and text such as "unknown", "N/A" or "not provided" are not always interpreted as missing automatically.

### 5.1 Inspect distinct values in a column

### 5.1. Inspect monthly spend values

```python
print(clean_df["monthly_spend"].unique())
```

**Expected output**

```text
Values including numeric-looking strings and 'unknown'
```

**Explanation**

Always inspect the actual source values before converting or replacing them.



### 5.2 Detect blank strings

### 5.2. Find blank or whitespace-only city values

```python
blank_city = clean_df["city"].astype("string").str.strip().eq("")

print(blank_city)
```

**Expected output**

```text
A Boolean Series indicating blank text values
```

**Explanation**

A blank string is not necessarily the same as a missing value. This check detects empty text after trimming whitespace.



### 5.3 Convert known markers into missing values

### 5.3. Normalize approved missing markers

```python
clean_df["monthly_spend"] = clean_df["monthly_spend"].replace(
    ["unknown", "Unknown", "N/A", ""],
    np.nan
)

print(clean_df["monthly_spend"])
```

**Expected output**

```text
The approved text markers are displayed as NaN
```

**Explanation**

Only replace values that the data definition confirms mean missing. Do not treat every word such as 'other' as missing.



### 5.4 Convert blank city strings to missing

### 5.4. Replace empty city values

```python
clean_df["city"] = clean_df["city"].replace(r"^\s*$", np.nan, regex=True)

print(clean_df["city"].isna().sum())
```

**Expected output**

```text
1 or more depending on source rows
```

**Explanation**

The regular expression matches strings containing only whitespace. Existing None values are also counted as missing.



### 5.5 Recheck missing counts

### 5.5. Review missingness after marker normalization

```python
print(clean_df.isna().sum())
```

**Expected output**

```text
Updated missing-value counts
```

**Explanation**

Always recheck after converting source markers. This confirms which missing values are now visible to Pandas.



## 6. Decide how to handle missing values

There is no single correct method for every column. The decision depends on the meaning of the field, how much data is missing and whether the value is needed for the prediction task.

Common options:
- Keep missingness and use a model or pipeline that supports it.
- Impute numeric values using a statistic such as median, fitted on training data.
- Impute categories using a known label or most frequent category, when justified.
- Remove rows when the target is missing and supervised training cannot use them.
- Remove a column only when it is not useful and the decision is justified.
- Ask the source owner to repair data if the missingness is unexpected.

### 6.1 Create a missingness report

### 6.1. Build a missing-value summary

```python
missing_report = pd.DataFrame({
    "missing_count": clean_df.isna().sum(),
    "missing_percent": (clean_df.isna().mean() * 100).round(2)
})

print(missing_report)
```

**Expected output**

```text
A table with missing counts and percentages for every column
```

**Explanation**

A report is more useful than immediately filling values because it makes the scale of the issue visible.



### 6.2 Fill a numeric example with the median

### 6.2. Impute age using the median for this demonstration

```python
age_median = clean_df["age"].median()
age_filled = clean_df["age"].fillna(age_median)

print(age_median)
print(age_filled)
```

**Expected output**

```text
The median age followed by the age values with the missing value filled
```

**Explanation**

This demonstrates the operation only. In a proper ML evaluation workflow, calculate the median from training data only and apply it to validation/test data using a fitted imputer or pipeline.



### 6.3 Fill a category using an explicit label

### 6.3. Create an explicit missing category

```python
city_filled = clean_df["city"].fillna("Not recorded")

print(city_filled)
```

**Expected output**

```text
A city Series with missing values shown as Not recorded
```

**Explanation**

This is appropriate only if preserving the missingness category makes sense for the task. The label must be consistent between training and inference.



### 6.4 Drop rows missing a required target

### 6.4. Remove rows with missing target values

```python
training_ready = clean_df.dropna(subset=["churn"]) if "churn" in clean_df.columns else clean_df.copy()

print(training_ready.shape)
```

**Expected output**

```text
The current shape, because this teaching dataset does not contain a churn column
```

**Explanation**

Supervised training requires known target labels. This dataset has no target, so the code keeps a copy. In a real dataset, confirm the target name before applying this operation.



### 6.5 Why not fill every missing numeric value with zero?

Zero can be a real value. Replacing a missing salary, age or transaction amount with zero creates an artificial observation and can distort the model. Use zero only when the field definition explicitly states that missing means zero.

## 7. Exact duplicate rows

An **exact duplicate** is a row whose values match another row across the columns being checked.

Duplicates may be caused by:
- Repeated file exports.
- Repeated ingestion.
- Joining tables incorrectly.
- Source-system retries.
- Legitimate repeated events that happen to share the same values.

### 7.1 Count exact duplicate rows

### 7.1. Use duplicated()

```python
print(clean_df.duplicated().sum())
```

**Expected output**

```text
The number of exact duplicate rows after the first occurrence
```

**Explanation**

By default, Pandas checks all columns and marks later copies as duplicates.



### 7.2 Display duplicate rows

### 7.2. Inspect duplicate records

```python
duplicate_rows = clean_df[clean_df.duplicated(keep=False)]

print(duplicate_rows)
```

**Expected output**

```text
All rows belonging to duplicate groups
```

**Explanation**

The keep=False option marks every member of a duplicate group. Inspect them before deleting anything.



### 7.3 Remove exact duplicates from a working copy

### 7.3. Drop exact duplicate rows

```python
deduplicated_df = clean_df.drop_duplicates()

print(deduplicated_df.shape)
```

**Expected output**

```text
A DataFrame with exact duplicate rows removed
```

**Explanation**

The operation returns a new DataFrame. Keep the raw data and record how many rows were removed.



### 7.4 Record the number of removed rows

### 7.4. Measure exact duplicate removal

```python
before_rows = len(clean_df)
after_rows = len(deduplicated_df)

print("Before:", before_rows)
print("After:", after_rows)
print("Removed:", before_rows - after_rows)
```

**Expected output**

```text
Before, after and removed row counts
```

**Explanation**

A cleaning step should be measurable. This makes it easier to explain the impact and audit the transformation.



## 8. Business-key duplicates

An exact duplicate check is not always enough. A dataset may have multiple rows with the same customer ID because one customer can make several purchases.

A **business-key duplicate** occurs when a key that is expected to be unique appears more than once at the intended row grain.

Examples:
- One row per customer: customer_id should be unique.
- One row per transaction: transaction_id should be unique.
- One row per customer per month: the combination of customer_id and month may be unique.

### 8.1 Check duplicate customer IDs

### 8.1. Find repeated customer IDs

```python
print(clean_df["customer_id"].duplicated().sum())
```

**Expected output**

```text
A count of repeated customer IDs
```

**Explanation**

Use this check only if the dataset is expected to contain one row per customer. If it is a transaction table, repeated customer IDs are expected.



### 8.2 Display repeated customer IDs

### 8.2. Inspect all repeated-ID rows

```python
repeated_ids = clean_df[
    clean_df["customer_id"].duplicated(keep=False)
].sort_values("customer_id")

print(repeated_ids)
```

**Expected output**

```text
Rows whose customer_id occurs more than once
```

**Explanation**

Sort the rows to compare their other fields. Determine whether the records are exact duplicates or separate legitimate events.



### 8.3 Check a composite key

### 8.3. Check customer and signup date combination

```python
key_columns = ["customer_id", "signup_date"]

print(clean_df.duplicated(subset=key_columns).sum())
```

**Expected output**

```text
The number of repeated customer_id and signup_date combinations
```

**Explanation**

Composite keys represent a combination of columns. Choose them from the business definition, not by trial and error.



### 8.4 Drop duplicates using a chosen key

### 8.4. Create a deduplicated customer view

```python
customer_view = clean_df.drop_duplicates(
    subset=["customer_id"],
    keep="first"
)

print(customer_view[["customer_id", "age", "city"]])
```

**Expected output**

```text
One retained row per customer ID
```

**Explanation**

This is only a demonstration. Keeping the first row is not automatically correct. If repeated rows contain different information, define a valid aggregation or source-of-truth rule.



## 9. Inconsistent text and categories

Inconsistent category labels can split one real category into several values.

Example:
- Hyderabad
- hyderabad
- HYDERABAD
- Hyderabad followed by a space

These may represent the same city, but confirm the allowed category list before normalizing.

### 9.1 Inspect unique city values

### 9.1. List distinct city labels

```python
print(clean_df["city"].dropna().unique())
```

**Expected output**

```text
Distinct non-missing city labels
```

**Explanation**

The output can reveal inconsistent capitalization, abbreviations and whitespace.



### 9.2 Remove leading and trailing whitespace

### 9.2. Strip city text

```python
clean_df["city"] = clean_df["city"].astype("string").str.strip()

print(clean_df["city"].unique())
```

**Expected output**

```text
City values without leading/trailing whitespace
```

**Explanation**

The `str.strip()` operation removes surrounding whitespace. Converting to Pandas string dtype helps preserve missing values as missing rather than converting them into the text 'None'.



### 9.3 Normalize category capitalization

### 9.3. Standardize case

```python
clean_df["city"] = clean_df["city"].str.title()

print(clean_df["city"].unique())
```

**Expected output**

```text
City labels with title-style capitalization
```

**Explanation**

This is suitable only when title case is an approved display convention. It does not resolve abbreviations or different names for the same city.



### 9.4 Map approved abbreviations

### 9.4. Map known abbreviations

```python
city_mapping = {
    "Hyd": "Hyderabad",
    "Vizag": "Visakhapatnam"
}

clean_df["city"] = clean_df["city"].replace(city_mapping)

print(clean_df["city"].unique())
```

**Expected output**

```text
Updated category values where an approved mapping exists
```

**Explanation**

Only use mappings confirmed by the data owner. Do not guess that two similar labels refer to the same place.



### 9.5 Recheck category frequencies

### 9.5. Count normalized city categories

```python
print(clean_df["city"].value_counts(dropna=False))
```

**Expected output**

```text
A count for each normalized city and missing category
```

**Explanation**

Check whether normalization combined only the values that were intended to represent the same category.



## 10. Convert numeric-looking text safely

A numeric column may be stored as text because it contains:
- Currency symbols.
- Commas.
- Units.
- Words such as unknown.
- Empty strings.
- Mixed decimal separators.

Do not cast the column blindly. Inspect values and define how invalid values should be treated.

### 10.1 Inspect the raw values

### 10.1. View monthly spend values

```python
print(clean_df["monthly_spend"].tolist())
```

**Expected output**

```text
The current monthly_spend values including numeric strings and missing markers
```

**Explanation**

This inspection helps identify the formatting patterns before conversion.



### 10.2 Convert approved numeric text

### 10.2. Use to_numeric with coercion

```python
clean_df["monthly_spend"] = pd.to_numeric(
    clean_df["monthly_spend"],
    errors="coerce"
)

print(clean_df["monthly_spend"])
```

**Expected output**

```text
Numeric values with unparseable entries represented as missing
```

**Explanation**

`errors='coerce'` converts invalid text to missing values. Count and investigate newly missing values; do not silently discard them.



### 10.3 Count invalid values after conversion

### 10.3. Count missing values after numeric conversion

```python
print(clean_df["monthly_spend"].isna().sum())
```

**Expected output**

```text
The number of missing or unparseable spend values
```

**Explanation**

Some missing values existed before conversion. Compare against the original data to identify which rows failed conversion.



### 10.4 Preserve an invalid-value audit

### 10.4. Identify values that could not be converted

```python
raw_spend = raw_df["monthly_spend"]

converted_spend = pd.to_numeric(
    raw_spend.replace(["unknown", "Unknown", "N/A", ""], np.nan),
    errors="coerce"
)

invalid_mask = raw_spend.notna() & converted_spend.isna()

print(raw_spend[invalid_mask])
```

**Expected output**

```text
The original non-missing text values that failed numeric conversion
```

**Explanation**

An audit lets the team inspect exactly what was changed. The list of approved missing markers must match the source definition.



## 11. Date parsing and inconsistent formats

Date columns often arrive as text. Correct parsing is important for time-based features, sorting and chronological splitting.

### 11.1 Inspect raw date values

### 11.1. Inspect signup dates

```python
print(clean_df["signup_date"].tolist())
```

**Expected output**

```text
The source date strings in their original forms
```

**Explanation**

The values may use different separators and ordering. Confirm whether a date such as 12-01-2026 means 12 January or 1 December.



### 11.2 Parse dates with a controlled rule

### 11.2. Parse the date column

```python
clean_df["signup_date"] = pd.to_datetime(
    clean_df["signup_date"],
    errors="coerce",
    dayfirst=True
)

print(clean_df["signup_date"])
```

**Expected output**

```text
Parsed datetime values with unparseable values shown as NaT
```

**Explanation**

The dayfirst setting is an explicit assumption for ambiguous formats. Confirm the source convention. Coercion should be followed by an invalid-date audit.



### 11.3 Inspect the resulting date dtype

### 11.3. Check the parsed date type

```python
print(clean_df["signup_date"].dtype)
```

**Expected output**

```text
datetime64[ns] (typical)
```

**Explanation**

Pandas uses a datetime dtype after successful parsing. Timezone-aware data may have a different dtype.



### 11.4 Extract year and month

### 11.4. Create date components

```python
clean_df["signup_year"] = clean_df["signup_date"].dt.year
clean_df["signup_month"] = clean_df["signup_date"].dt.month

print(clean_df[["signup_date", "signup_year", "signup_month"]])
```

**Expected output**

```text
Date values with year and month columns
```

**Explanation**

Date components can be useful features when they are available at prediction time and reflect a valid business relationship.



## 12. Validate numeric ranges and business rules

A value can be numeric but still invalid. For example, age 150 may be suspicious, and a negative order quantity may violate a business rule.

Do not invent universal limits. Ask the domain owner for valid ranges.

### 12.1 Inspect age range

### 12.1. Check minimum and maximum age

```python
print(clean_df["age"].min())
print(clean_df["age"].max())
```

**Expected output**

```text
The minimum and maximum values currently present
```

**Explanation**

Range summaries can reveal suspicious values. They do not prove which values are invalid.



### 12.2 Flag suspicious ages using an approved rule

### 12.2. Create a review flag

```python
age_review = (clean_df["age"] < 0) | (clean_df["age"] > 120)

print(clean_df.loc[age_review, ["customer_id", "age"]])
```

**Expected output**

```text
The row with age 150 is flagged for review
```

**Explanation**

The range is illustrative. A production rule must be agreed with the domain owner and may vary by use case.



### 12.3 Do not delete a suspicious record automatically

First verify:
- Could the value be a data-entry error?
- Is the age measured in years or months?
- Is it a synthetic or test record?
- Is the customer record incorrectly joined?
- Does the value belong to a different entity?

## 13. Choosing a cleaning action

| Problem | Possible action | What to verify |
|---|---|---|
| Missing numeric feature | Impute or preserve missingness | Training-only fitting and missingness meaning |
| Missing target | Exclude from supervised fitting or obtain label | Whether target can be recovered |
| Exact duplicate row | Remove duplicate copy | Whether duplicate is truly accidental |
| Repeated entity ID | Aggregate, retain events or correct join | Observation unit |
| Inconsistent category | Normalize using approved mapping | Category definitions |
| Invalid numeric text | Parse, mark missing or reject | Original value and business rule |
| Invalid date | Correct source or mark invalid | Date format and timezone |
| Extreme numeric value | Investigate and decide | Validity and model sensitivity |

## 14. Keep a cleaning audit

A cleaning audit records what changed and why. It supports debugging, reproducibility and stakeholder review.

### 14.1 Record original row count

### 14.1. Store the original row count

```python
original_row_count = len(raw_df)

print(original_row_count)
```

**Expected output**

```text
7
```

**Explanation**

Keep the original count so you can reconcile later transformations.



### 14.2 Record duplicate count before removal

### 14.2. Store duplicate count

```python
duplicate_count = raw_df.duplicated().sum()

print(duplicate_count)
```

**Expected output**

```text
A duplicate count based on all columns
```

**Explanation**

This is one audit metric. Business-key duplicates should be tracked separately.



### 14.3 Create a simple audit dictionary

### 14.3. Record cleaning decisions

```python
cleaning_audit = {
    "original_rows": len(raw_df),
    "exact_duplicates": int(raw_df.duplicated().sum()),
    "missing_age": int(raw_df["age"].isna().sum()),
    "missing_city": int(raw_df["city"].isna().sum())
}

print(cleaning_audit)
```

**Expected output**

```text
A dictionary containing initial row and missing-value counts
```

**Explanation**

This is a simple in-memory audit example. Production pipelines should persist structured logs and the rules applied.



### 14.4 Compare before and after

### 14.4. Compare data shapes

```python
print("Raw shape:", raw_df.shape)
print("Working shape:", clean_df.shape)
```

**Expected output**

```text
The original and working DataFrame shapes
```

**Explanation**

A shape comparison helps identify whether rows or columns were added or removed.



## 15. Validate after cleaning

Cleaning is not complete until the output is checked.

### 15.1 Check remaining missing values

### 15.1. Review missing values again

```python
print(clean_df.isna().sum())
```

**Expected output**

```text
A missing-value count for every column
```

**Explanation**

Some missing values may be intentionally preserved. Explain why each remaining missing value is acceptable or how a later pipeline will handle it.



### 15.2 Check duplicate rows again

### 15.2. Recheck exact duplicates

```python
print(clean_df.duplicated().sum())
```

**Expected output**

```text
The remaining exact duplicate count
```

**Explanation**

This verifies whether the chosen duplicate operation had the intended effect.



### 15.3 Check data types again

### 15.3. Review cleaned dtypes

```python
print(clean_df.dtypes)
```

**Expected output**

```text
Updated column data types
```

**Explanation**

Confirm numeric columns are numeric, dates are datetime and categories have the expected representation.



### 15.4 Check row count and record loss

### 15.4. Compare row counts

```python
print("Raw rows:", len(raw_df))
print("Current rows:", len(clean_df))
print("Difference:", len(raw_df) - len(clean_df))
```

**Expected output**

```text
The number of original rows, current rows and difference
```

**Explanation**

A row-count change should be explainable. Never lose records silently.



### 15.5 Check business-key uniqueness only if required

### 15.5. Validate unique customer IDs for a one-row-per-customer table

```python
if clean_df["customer_id"].is_unique:
    print("Customer IDs are unique")
else:
    print("Repeated customer IDs need review")
```

**Expected output**

```text
A message indicating whether customer IDs are unique
```

**Explanation**

Use this rule only when one row per customer is the approved grain. It is not valid for transaction or customer-month tables.



## 16. Common beginner mistakes

| Mistake | Why it is risky | Better practice |
|---|---|---|
| Replace all missing values with zero | Zero may be a valid measurement | Define meaning and choose a justified strategy |
| Drop every row with any missing value | Can remove too much data or bias the sample | Inspect columns and missingness patterns |
| Remove repeated customer IDs | Customers may legitimately have multiple transactions | Confirm row grain and business key |
| Use drop_duplicates without review | May remove valid events | Inspect duplicate groups first |
| Convert every object column to numeric | Categories and IDs are not measurements | Inspect semantics before conversion |
| Use errors="coerce" without an audit | Invalid text becomes missing silently | Count and inspect coerced values |
| Normalize categories without mapping rules | Distinct categories may be merged incorrectly | Use an approved mapping |
| Parse ambiguous dates without confirming format | Day and month can be swapped | Confirm source convention |
| Clean before preserving raw data | Original values become difficult to recover | Keep immutable raw data and a working copy |
| Skip post-cleaning validation | New problems may be introduced | Recheck types, missing values, duplicates and row counts |

## 17. ML-specific cleaning principles

1. Split the data appropriately before fitting imputers or scalers used for model evaluation.
2. Fit learned preprocessing steps using training data only.
3. Apply the same fitted transformation to validation, test and production data.
4. Do not use information recorded after the prediction time.
5. Keep feature and target rows aligned.
6. Preserve identifiers for traceability where appropriate.
7. Document every removal, replacement and conversion.
8. Treat cleaning rules as part of the model pipeline, not an undocumented notebook action.

## 18. Interview-ready explanations

**What is data cleaning?**  
Data cleaning identifies and handles incomplete, duplicated, inconsistent or invalid values so that analysis and modelling use reliable information. Each cleaning decision should be supported by the data definition and business context.

**What is the difference between missing and zero?**  
Missing means information is unavailable or not recorded. Zero is a numeric value that may be meaningful. Replacing missing with zero can change the meaning of the data.

**How do you handle duplicate records?**  
First identify the observation unit and business key. Then inspect duplicate groups to distinguish accidental repeated records from legitimate repeated events. Remove or aggregate only according to an approved rule.

**Why should an ML engineer preserve raw data?**  
It allows the team to audit transformations, reproduce cleaning, compare before-and-after results and recover from an incorrect rule.

**Why is fitting an imputer on the full dataset a leakage risk?**  
The imputer learns information such as a median from held-out rows. Fit it on training data only and apply the learned transformation to held-out data.

## 19. Scenario questions

1. A missing salary is replaced with zero. What is the concern?  
   **Answer:** Zero may represent an actual salary value and is not equivalent to missing information. Use a justified missing-data strategy.

2. A customer appears 12 times in the dataset. Should you remove 11 rows?  
   **Answer:** Not until the row grain is known. The data may represent monthly activity or separate transactions.

3. A numeric column contains "unknown". What do you do?  
   **Answer:** Confirm the meaning, record the affected values, convert valid numbers and apply an approved rule to the unknown entries.

4. After cleaning, the dataset loses 30% of its rows. What should you do?  
   **Answer:** Stop and investigate which rules removed records, why those rows were affected and whether the resulting sample is biased.

5. A date parser changes 03-04-2026 to an unexpected date. What do you check?  
   **Answer:** Confirm whether the source uses day-first or month-first format and inspect the original source convention.

## 20. Lesson completion checklist

- [ ] I can explain missing values, blank strings and zero.
- [ ] I can count missing values by column and row.
- [ ] I can inspect and handle special missing markers.
- [ ] I can detect exact duplicates.
- [ ] I understand business-key duplicates.
- [ ] I can normalize categories using an approved mapping.
- [ ] I can convert numeric text safely.
- [ ] I can parse dates using a confirmed format.
- [ ] I can record cleaning decisions.
- [ ] I can validate the dataset after cleaning.
- [ ] I understand why learned preprocessing must be fitted on training data only.

**Next lesson:** Day 11 — Outlier Detection and Treatment.
