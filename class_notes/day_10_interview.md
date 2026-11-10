# Day 10: Interview Questions — Missing Values, Duplicates and Inconsistent Data

**Pathway:** Machine Learning Engineer  
**Focus:** Data quality, safe cleaning decisions and validation

## 1. Core concepts

### Q1. What is data cleaning and why is it important in ML?

**Answer:** Data cleaning identifies and handles incomplete, duplicated, inconsistent or invalid data so that analysis and models use trustworthy inputs. Poor-quality data can create misleading patterns or cause a pipeline to fail.

**Follow-up:** Does cleaning mean removing every unusual value?

**Answer:** No. An unusual value may be valid. Investigate its meaning before changing it.

### Q2. What is the difference between missing and zero?

**Answer:** Missing means information is unavailable or not recorded. Zero is a numeric value that may be meaningful. Replacing missing with zero can change the meaning of the data.

**Follow-up:** Give an example.

**Answer:** A missing monthly salary is not the same as a recorded salary of zero.

### Q3. What is the difference between a blank string and NaN?

**Answer:** A blank string is text containing no visible characters, while NaN is a missing-value marker recognized by Pandas for many operations. A CSV reader may not interpret every blank or special text marker as missing automatically.

**Follow-up:** How would you inspect this?

**Answer:** Inspect distinct values, strip whitespace where appropriate and explicitly normalize approved missing markers.

## 2. Missing-value scenarios

### Q4. How do you check missing values in Pandas?

**Answer:** Use `df.isna().sum()` for counts by column. Use `df.isna().mean() * 100` for percentages, and filter rows with missing values to inspect affected records.

**Follow-up:** Does an overall missing-value count tell you which records are affected?

**Answer:** No. Inspect row-level masks or column-specific filters.

### Q5. How would you handle missing values in a numeric feature?

**Answer:** First inspect why values are missing, the proportion affected and whether missingness has business meaning. Possible approaches include median/mean imputation, a missing indicator, a model that supports missing values or removing a column/record when justified. For model evaluation, fit imputation using training data only.

**Follow-up:** Why might median imputation be chosen over mean?

**Answer:** Median is less sensitive to extreme values and may be more representative for a skewed feature.

### Q6. Should you drop every row with a missing value?

**Answer:** No. It can remove a large share of data and create selection bias. Check which columns are missing, how many records are affected and whether the remaining data still represents the target population.

**Follow-up:** When might dropping a row be justified?

**Answer:** For example, a supervised training row without a known target cannot be used for ordinary supervised fitting, provided the exclusion is documented and appropriate.

### Q7. You use errors="coerce" during numeric conversion. What must you do next?

**Answer:** Count the values converted to missing, inspect the original values, determine why conversion failed and apply an approved cleaning rule. Coercion should not silently hide source-data problems.

**Follow-up:** Why keep a copy of the original column?

**Answer:** It allows an audit of which values changed and supports investigation or recovery.

## 3. Duplicate and inconsistency scenarios

### Q8. What is the difference between exact duplicates and business-key duplicates?

**Answer:** Exact duplicates match across the selected columns. Business-key duplicates repeat a key that should be unique at the intended row grain. The same customer ID can legitimately appear multiple times in a transaction table.

**Follow-up:** What should you establish before using drop_duplicates?

**Answer:** The observation unit, uniqueness key and rule for choosing or combining repeated records.

### Q9. A customer appears 12 times. Should you keep only one row?

**Answer:** Not until the row grain is clear. The rows may represent 12 transactions or 12 monthly observations. Removing them could destroy valid history.

**Follow-up:** How would you check?

**Answer:** Inspect transaction IDs, timestamps, event types and the source-table definition.

### Q10. City values appear as Hyderabad, hyderabad and HYD. What would you do?

**Answer:** Inspect the distinct values and confirm which labels represent the same city. Trim whitespace and apply an approved mapping. Then recheck category counts.

**Follow-up:** Why not simply convert everything to title case?

**Answer:** It can normalize capitalization but cannot resolve abbreviations, spelling differences or genuinely distinct categories.

### Q11. A date column contains both 03-04-2026 and 2026/04/03. What is the risk?

**Answer:** The first value is ambiguous without a known day-first or month-first convention. Incorrect parsing can swap dates and corrupt chronological analysis or train-test splitting.

**Follow-up:** How would you handle it?

**Answer:** Confirm the source format and timezone, parse with an explicit rule, then inspect invalid values and sample conversions.

### Q12. After cleaning, 30% of the records disappear. What would you do?

**Answer:** Stop and reconcile row counts. Identify which cleaning rules removed the records, inspect affected segments and determine whether removal creates bias. Revise the rule or seek source correction if necessary.

**Follow-up:** Why is an audit log useful?

**Answer:** It records the number of changes, the rule applied and the reason, supporting reproducibility and review.

## 4. ML pipeline scenarios

### Q13. Why should an imputer be fitted on training data only?

**Answer:** The imputer learns values such as a median from the data. If it is fitted on the full dataset, validation or test information influences preprocessing and makes evaluation less independent. Fit it on training data and apply the fitted transformer to held-out data.

**Follow-up:** How can a pipeline help?

**Answer:** A scikit-learn Pipeline keeps preprocessing and model fitting together so each training fold fits its own transformations.

### Q14. A feature is missing in production but was complete during training. What should you do?

**Answer:** Check whether the production source changed, validate the input schema, measure the missingness rate and use the documented preprocessing strategy. If the change is unexpected or outside the supported contract, alert the owner and consider rejecting the request rather than silently producing unreliable predictions.

**Follow-up:** Should production use a different imputation rule from training?

**Answer:** No. It should apply the same fitted transformation and approved inference behaviour.

### Q15. Why preserve the raw dataset?

**Answer:** It provides an immutable reference for auditing, reproducing transformations and recovering from incorrect cleaning decisions. The cleaned dataset should be a derived artifact.

**Follow-up:** Should raw customer data be copied freely into personal folders?

**Answer:** No. Follow data access, privacy and security policies. Preserve only authorized data in approved storage.

## 5. Code interpretation questions

### Q16. What does this code count?

```python
df.duplicated().sum()
```

**Answer:** The number of duplicate rows after the first occurrence, using all columns by default.

**Follow-up:** How do you mark all members of duplicate groups?

**Answer:** Use `df.duplicated(keep=False)`.

### Q17. What does this code do?

```python
pd.to_numeric(df["amount"], errors="coerce")
```

**Answer:** It converts parseable values to numeric and returns missing values for entries that cannot be converted.

**Follow-up:** Does it modify the original column automatically?

**Answer:** Not unless the result is assigned back to the DataFrame column.

### Q18. What is the purpose of df.copy() before cleaning?

**Answer:** It creates a working copy so the original DataFrame can remain unchanged for comparison and audit.

**Follow-up:** Is a shallow DataFrame copy always an independent deep copy of every nested Python object?

**Answer:** No. Pandas copy behaviour depends on the data and copy semantics; for ordinary tabular scalar columns it is commonly used as a separate working DataFrame, but nested mutable objects require additional care.

## 6. Interview answer framework

1. Preserve the raw source.
2. Profile missingness, duplicates and inconsistent values.
3. Confirm row grain and business definitions.
4. Choose the smallest justified cleaning action.
5. Record before-and-after counts.
6. Fit learned preprocessing on training data only.
7. Validate types, missingness, uniqueness and row alignment after cleaning.

**Senior developer principle:** A cleaning operation is not successful merely because the code ran. It is successful when the transformation is justified, measurable, reproducible and validated.
