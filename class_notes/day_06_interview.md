# Day 06: Interview Questions — Pandas Series, DataFrames and File Reading

**Pathway:** Machine Learning Engineer  
**Focus:** Tabular data inspection, file loading, selection and ML data validation

## 1. Core concepts

### Q1. What is Pandas and why is it used in machine learning?

**Answer:** Pandas is a Python library for working with labelled tabular data. It provides Series and DataFrame structures, file-reading functions, selection, filtering, grouping and data-inspection operations. It is widely used before model training.

**Follow-up:** Is Pandas a machine-learning algorithm?

**Answer:** No. It is a data manipulation and analysis library often used to prepare data for ML.

### Q2. What is the difference between a Series and a DataFrame?

**Answer:** A Series is a one-dimensional labelled collection. A DataFrame is a two-dimensional labelled table with rows and columns, and different columns can have different data types.

**Follow-up:** What does `df["age"]` return?

**Answer:** Usually a Series. `df[["age"]]` returns a DataFrame with one column.

### Q3. What is the role of the DataFrame index?

**Answer:** The index labels rows and helps identify or align them during selection and operations. It is not automatically a unique business identifier.

**Follow-up:** Should you assume the index is a customer ID?

**Answer:** No. Inspect the index and dataset schema. A default RangeIndex is simply a row position label.

## 2. File reading questions

### Q4. How do you read a CSV file in Pandas?

**Answer:** Use `pd.read_csv("file.csv")`. Confirm the file path, delimiter, encoding and relevant data types. Inspect the resulting DataFrame before analysis.

**Follow-up:** What if the file uses semicolons?

**Answer:** Use `sep=";"` after confirming the file's delimiter.

### Q5. What is the purpose of the usecols parameter?

**Answer:** It loads only selected columns from the file. This can reduce memory use and limit the data to fields required for the task.

**Follow-up:** What if a requested column does not exist?

**Answer:** The read operation may fail. Inspect the source header and confirm the exact column names.

### Q6. What is the purpose of nrows?

**Answer:** It limits how many rows are read. It is useful for quick inspection, but a small sample may not reveal all categories, missing-value patterns or rare data problems.

**Follow-up:** Can you use a 10-row sample to certify the full dataset is clean?

**Answer:** No. A sample is useful for initial inspection, not complete validation.

### Q7. What do you do if Pandas cannot decode a CSV file?

**Answer:** Identify the file's actual encoding from the source or data owner. Then pass the correct encoding to the reader. Do not repeatedly guess encodings without checking, because incorrect decoding may corrupt text.

**Follow-up:** Is UTF-8 always guaranteed?

**Answer:** No. It is common, but files can use other encodings.

### Q8. What do you do if the file path is not found?

**Answer:** Check the current working directory, inspect the folder structure, verify spelling and file extension, and use a correct relative or absolute path. Confirm the file exists before attempting to read it.

**Follow-up:** Does the relative path always start from the notebook's folder?

**Answer:** Not necessarily. It is resolved from the process's current working directory, which should be inspected.

### Q9. How do you read an Excel file?

**Answer:** Use `pd.read_excel("file.xlsx")`. A compatible optional engine may be required depending on the workbook format and environment. Use `sheet_name` to select a worksheet.

**Follow-up:** What if the worksheet name is incorrect?

**Answer:** Confirm the workbook's sheet names and pass the exact intended name.

## 3. Dataset inspection questions

### Q10. How do you inspect a new dataset before model training?

**Answer:** I inspect `head()`, `tail()`, `shape`, `columns`, `dtypes`, `info()` and `describe()`. I also check missing values, duplicate records, category frequencies, target distribution and the business meaning of each field.

**Follow-up:** Does head() prove the dataset is clean?

**Answer:** No. It shows only a small number of rows. A full inspection needs structural, statistical and domain-specific checks.

### Q11. What does df.info() tell you?

**Answer:** It reports the index range, column names, non-null counts, data types and memory usage. It helps identify missing values and unexpected data types.

**Follow-up:** Does info() identify all invalid values?

**Answer:** No. A column can have the expected dtype but still contain invalid values or incorrect units.

### Q12. What does df.describe() provide?

**Answer:** It provides summary statistics for columns, such as count, mean, standard deviation, minimum, quartiles and maximum for numeric data. With suitable options, it can summarize non-numeric columns too.

**Follow-up:** Can a suspiciously high maximum be declared an error from describe() alone?

**Answer:** No. It is a signal to investigate. The value may be valid, an outlier or a data-entry issue depending on context.

### Q13. How do you check missing values?

**Answer:** Use `df.isnull().sum()` to count missing values by column. You can calculate percentages and inspect affected records before deciding how to handle them.

**Follow-up:** Should you drop every row containing a missing value?

**Answer:** No. First inspect the missingness pattern, its meaning, the number of affected records and the impact of removing them.

## 4. Selection and filtering

### Q14. What is the difference between loc and iloc?

**Answer:** `loc` selects by index and column labels. `iloc` selects by integer position.

**Follow-up:** What does `df.iloc[0]` select?

**Answer:** The first row by position, regardless of the row's label.

### Q15. How do you filter rows where age is above 30 and income is above 50,000?

**Answer:**
```python
df[(df["age"] > 30) & (df["income"] > 50000)]
```
Each condition is parenthesized and `&` combines the Boolean masks element by element.

**Follow-up:** Why not use Python's and?

**Answer:** A Pandas Series contains many Boolean values. Scalar `and` expects one truth value; `&` performs element-wise combination.

### Q16. What is isin() used for?

**Answer:** `isin()` checks whether each value belongs to a supplied collection. It is useful for filtering records that match one of several categories.

**Follow-up:** How is it different from checking equality with one value?

**Answer:** Equality checks one comparison value; `isin()` checks membership in multiple candidate values.

### Q17. How do you sort a DataFrame?

**Answer:** Use `sort_values(by="column_name")`. Set `ascending=False` for descending order. Multiple columns can be supplied when the sort has more than one key.

**Follow-up:** Does sort_values() necessarily change the original DataFrame?

**Answer:** By default it returns a sorted result. Assign the result if you want to keep it in a variable.

## 5. ML data scenarios

### Q18. A numerical feature is loaded with dtype object. What do you do?

**Answer:** Inspect its actual values. Look for currency symbols, commas, text markers, inconsistent formats and missing-value codes. Confirm the business meaning, clean valid representations and convert carefully. If using `pd.to_numeric(errors="coerce")`, count and investigate the values that became missing.

**Follow-up:** Should every object column be converted to a number?

**Answer:** No. Object columns may contain categories, identifiers or text. Conversion depends on the meaning of the field.

### Q19. A CSV file is expected to have 10,000 rows, but Pandas loads 9,800. How would you investigate?

**Answer:** Confirm whether the expected count includes a header or blank lines. Inspect the source file, delimiter, quoting, malformed rows and parser warnings. Check whether filters or parameters were applied and compare row counts at each processing stage. Do not silently assume the missing 200 rows are irrelevant.

**Follow-up:** Why is row-count reconciliation important?

**Answer:** It can reveal data loss, parser issues or unintended filtering before analysis and model training.

### Q20. The customer ID column contains values such as "00123". Why might it need to be read as text?

**Answer:** An ID is a label, not necessarily a quantity. Reading it as an integer may remove leading zeros and change the identifier. Use a string dtype when preserving the exact identifier representation is required.

**Follow-up:** Should customer ID automatically be used as a model feature?

**Answer:** No. Its predictive meaning and leakage risk must be evaluated. Often it is an identifier for tracking and joining rather than a meaningful feature.

### Q21. A target column has values 0, 1 and "unknown". What do you do?

**Answer:** Verify the target definition with the data owner. Inspect the frequency of each value and determine whether "unknown" means missing, a separate class or an invalid record. Do not silently convert it to zero or drop it without an approved rule.

**Follow-up:** Why is target cleaning especially important?

**Answer:** Incorrect target labels teach the model the wrong outcome and can distort evaluation.

### Q22. You create a new column by multiplying monthly charges by 12. What assumption are you making?

**Answer:** I am assuming the monthly charge is representative for all 12 months and that the business meaning of the derived value is annualized charges. I would verify whether the charge varies over time, whether discounts or partial months apply and whether the derived feature is valid for the prediction date.

**Follow-up:** Could this feature cause leakage?

**Answer:** It could if it uses information from after the prediction time or data unavailable when the prediction is made. Feature availability must be checked against the prediction timestamp.

### Q23. You separate X and y from a DataFrame. What should you validate?

**Answer:** Confirm that the intended feature columns and target column exist, that the target is correctly defined, that the feature and target rows remain aligned, and that no target or post-outcome information is accidentally included in X.

**Follow-up:** Are equal row counts enough?

**Answer:** No. The observations and their order must correspond correctly.

## 6. Code interpretation questions

### Q24. What is the output type?

```python
age_series = df["age"]
one_column_df = df[["age"]]
```

**Answer:** `age_series` is a Series; `one_column_df` is a DataFrame.

**Follow-up:** Why can this distinction matter?

**Answer:** Different methods and downstream APIs may expect a one-dimensional Series or a two-dimensional feature table.

### Q25. What does this expression return?

```python
df.loc[0, ["age", "churn"]]
```

**Answer:** The age and churn values from the row with index label 0, if that label exists.

**Follow-up:** How would you select the first row by position?

**Answer:** Use `iloc[0]`.

### Q26. What does this code do?

```python
df["high_charge"] = df["monthly_charges"] > 500
```

**Answer:** It creates a Boolean column indicating whether each row's monthly_charges value is greater than 500.

**Follow-up:** What should you inspect before running it?

**Answer:** Confirm the column exists, contains appropriate numeric values and that the threshold has a valid business interpretation.

## 7. Interview answer framework

When asked about a dataset-loading or Pandas scenario:

1. **Confirm the source:** File path, format, version and expected schema.
2. **Load carefully:** Use correct delimiter, encoding, worksheet and data-type options.
3. **Inspect structure:** Shape, columns, dtypes, info and sample rows.
4. **Validate meaning:** IDs, units, categories, target and observation grain.
5. **Check quality:** Missing values, duplicates, invalid values and unexpected categories.
6. **Preserve alignment:** Keep features and target rows associated correctly.
7. **Record decisions:** Document parsing options, rejected rows and assumptions.

**Senior developer principle:** A DataFrame loading without an error does not prove that the data was interpreted correctly. Validate both the technical schema and the business meaning before using it for ML.
