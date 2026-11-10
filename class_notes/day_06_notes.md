# Day 06: Pandas Series, DataFrames and File Reading

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, Mathematics and Statistics Foundations  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

By the end of this lesson, you should be able to:

- Explain why Pandas is useful for tabular data.
- Import Pandas using the common alias.
- Create and inspect a Series.
- Create a DataFrame from dictionaries and lists.
- Understand rows, columns, index labels and data types.
- Read CSV files and understand common reading parameters.
- Read Excel files when the required engine is available.
- Inspect a dataset using `head()`, `tail()`, `shape`, `columns`, `dtypes`, `info()` and `describe()`.
- Select columns and rows using label-based and position-based selection.
- Filter, sort and create simple columns.
- Recognize file-path, delimiter, encoding and data-type issues.
- Explain how Pandas supports machine-learning data preparation.

## 2. Why do we need Pandas?

A real dataset is usually stored as a table: each row represents an observation, and each column represents a field or feature. Python lists can store values, and NumPy can process arrays, but Pandas provides labelled tabular structures that make data inspection and manipulation easier.

### 2.1 Example of a business dataset

| customer_id | age | monthly_charges | contract_type | churn |
|---|---:|---:|---|---:|
| C001 | 25 | 450.50 | Monthly | 0 |
| C002 | 34 | 520.00 | Annual | 1 |
| C003 | 29 | 610.75 | Monthly | 0 |

In this example:
- Each row represents one customer.
- Each column stores one field.
- `churn` is the target column in this illustrative supervised-learning dataset.
- Other columns may be used as features after suitable validation and preprocessing.

## 3. Import Pandas

### 3.1 Import the library

### 3.1. Import Pandas

```python
import pandas as pd

print(pd.__version__)
```

**Expected output**

```text
A version string such as 2.x.x
```

**Explanation**

The alias `pd` is the standard convention. The exact version depends on the environment. Record package versions when reproducibility matters.



## 4. What is a Pandas Series?

A Series is a one-dimensional labelled collection of values. It contains values and an associated index.

### 4.1 Create a Series from a list

### 4.1. Create a simple Series

```python
import pandas as pd

ages = pd.Series([25, 34, 29, 41])

print(ages)
```

**Expected output**

```text
0    25
1    34
2    29
3    41
dtype: int64
```

**Explanation**

Pandas creates a default index beginning at zero. The left side shows the index; the right side shows the value.



### 4.2 Create a Series with custom labels

### 4.2. Provide index labels

```python
import pandas as pd

ages = pd.Series(
    [25, 34, 29],
    index=["C001", "C002", "C003"]
)

print(ages)
```

**Expected output**

```text
C001    25
C002    34
C003    29
dtype: int64
```

**Explanation**

The `index` argument supplies labels for the values. Here, customer IDs are used as labels.



### 4.3 Read one value by label

### 4.3. Select a Series value by label

```python
import pandas as pd

ages = pd.Series([25, 34, 29], index=["C001", "C002", "C003"])

print(ages["C002"])
```

**Expected output**

```text
34
```

**Explanation**

The label `C002` retrieves its corresponding value.



### 4.4 Read one value by position

### 4.4. Select a Series value by position

```python
import pandas as pd

ages = pd.Series([25, 34, 29], index=["C001", "C002", "C003"])

print(ages.iloc[1])
```

**Expected output**

```text
34
```

**Explanation**

`iloc` selects by integer position. Position one means the second value.



### 4.5 Inspect Series properties

### 4.5. Check Series values and metadata

```python
import pandas as pd

ages = pd.Series([25, 34, 29], index=["C001", "C002", "C003"])

print(ages.index)
print(ages.values)
print(ages.dtype)
print(ages.shape)
```

**Expected output**

```text
Index(['C001', 'C002', 'C003'], dtype='object')
[25 34 29]
int64
(3,)
```

**Explanation**

The index contains labels, values exposes the underlying values, dtype describes the value type, and shape shows the number of elements.



## 5. What is a DataFrame?

A DataFrame is a two-dimensional labelled table with rows and columns. Different columns can have different data types.

### 5.1 Create a DataFrame from a dictionary

### 5.1. Create a customer DataFrame

```python
import pandas as pd

customer_data = {
    "customer_id": ["C001", "C002", "C003"],
    "age": [25, 34, 29],
    "monthly_charges": [450.50, 520.00, 610.75],
    "churn": [0, 1, 0]
}

df = pd.DataFrame(customer_data)

print(df)
```

**Expected output**

```text
  customer_id  age  monthly_charges  churn
0        C001   25           450.50      0
1        C002   34           520.00      1
2        C003   29           610.75      0
```

**Explanation**

Dictionary keys become column names. Each list supplies the values for that column. The lists must have compatible lengths.



### 5.2 Create a DataFrame from a list of dictionaries

### 5.2. Represent each row as a dictionary

```python
import pandas as pd

records = [
    {"customer_id": "C001", "age": 25, "churn": 0},
    {"customer_id": "C002", "age": 34, "churn": 1},
    {"customer_id": "C003", "age": 29, "churn": 0}
]

df = pd.DataFrame(records)

print(df)
```

**Expected output**

```text
  customer_id  age  churn
0        C001   25      0
1        C002   34      1
2        C003   29      0
```

**Explanation**

Each dictionary represents a row. Pandas aligns values using the keys as column names.



### 5.3 Create a DataFrame from a list of lists

### 5.3. Provide column names separately

```python
import pandas as pd

rows = [
    ["C001", 25, 0],
    ["C002", 34, 1],
    ["C003", 29, 0]
]

df = pd.DataFrame(rows, columns=["customer_id", "age", "churn"])

print(df)
```

**Expected output**

```text
  customer_id  age  churn
0        C001   25      0
1        C002   34      1
2        C003   29      0
```

**Explanation**

The `columns` argument provides labels for the values in each row. The order of the column names must match the order of the row values.



### 5.4 Inspect the DataFrame type

### 5.4. Check the object type

```python
print(type(df))
```

**Expected output**

```text
<class 'pandas.core.frame.DataFrame'>
```

**Explanation**

This confirms that `df` is a DataFrame.



## 6. Understand rows, columns and index

### 6.1 Read the column names

### 6.1. Inspect columns

```python
print(df.columns)
```

**Expected output**

```text
Index(['customer_id', 'age', 'churn'], dtype='object')
```

**Explanation**

The `columns` attribute returns the DataFrame's column labels.



### 6.2 Inspect the row index

### 6.2. Inspect the index

```python
print(df.index)
```

**Expected output**

```text
RangeIndex(start=0, stop=3, step=1)
```

**Explanation**

Pandas created a default integer index for the three rows.



### 6.3 Inspect the shape

### 6.3. Check row and column counts

```python
print(df.shape)
```

**Expected output**

```text
(3, 3)
```

**Explanation**

The first number is the row count; the second is the column count. Shape is an attribute, so do not add parentheses.



### 6.4 Count all elements

### 6.4. Check total DataFrame values

```python
print(df.size)
```

**Expected output**

```text
9
```

**Explanation**

There are three rows and three columns, so the table contains nine cells.



### 6.5 Count rows

### 6.5. Use len()

```python
print(len(df))
```

**Expected output**

```text
3
```

**Explanation**

For a DataFrame, `len(df)` returns the number of rows.



## 7. Read CSV files

CSV means comma-separated values. It is a common format for sharing tabular data.

### 7.1 Understand the file path

A file path tells Python where the file is stored. Relative paths are interpreted from the notebook or script's working directory. Absolute paths specify the full location.

Example path:

```text
data/customer_churn.csv
```

A common project layout is:

```text
ml_project/
├── data/
│   └── customer_churn.csv
├── notebooks/
│   └── day_06_pandas.ipynb
└── src/
```

If the notebook is inside `notebooks/`, the relative path may need to be `../data/customer_churn.csv`. Always confirm the current working directory.

### 7.2 Read a CSV file

### 7.1. Load a CSV into a DataFrame

```python
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")

print(df.head())
```

**Expected output**

```text
The first five rows of the CSV file, assuming the file exists at the supplied path.
```

**Explanation**

`read_csv()` reads the file and returns a DataFrame. The path must match the actual location. If the file is missing, Pandas raises a file-related error.



### 7.3 Read a CSV from the current directory

### 7.2. Use a file in the current folder

```python
import pandas as pd

df = pd.read_csv("customer_churn.csv")

print(df.shape)
```

**Expected output**

```text
The row and column counts of the loaded file.
```

**Explanation**

This works only if the file is in the process's current working directory. Do not assume the notebook file's folder is always the current working directory.



### 7.4 Inspect the working directory

### 7.3. Check the current working directory

```python
import os

print(os.getcwd())
```

**Expected output**

```text
A path showing the current working directory.
```

**Explanation**

The output helps explain why a relative file path may or may not resolve. The path depends on the environment.



### 7.5 Check whether a file exists

### 7.4. Verify the file path

```python
import os

file_path = "data/customer_churn.csv"

print(os.path.exists(file_path))
```

**Expected output**

```text
True if the file exists at that path; otherwise False.
```

**Explanation**

This is a simple pre-check. It does not confirm that the file is readable or contains valid data.



## 8. Important read_csv() parameters

### 8.1 sep

Use `sep` when the file uses a delimiter other than a comma.

### 8.1. Read a semicolon-separated file

```python
import pandas as pd

df = pd.read_csv("data/customer_churn.csv", sep=";")

print(df.head())
```

**Expected output**

```text
The first five rows if the source file is semicolon-separated.
```

**Explanation**

Do not change the separator randomly. Inspect the file or confirm its format first.



### 8.2 nrows

Read only a limited number of rows when checking a large file.

### 8.2. Read the first ten rows

```python
import pandas as pd

sample_df = pd.read_csv("data/customer_churn.csv", nrows=10)

print(sample_df.shape)
```

**Expected output**

```text
A shape with ten or fewer rows and the file's column count.
```

**Explanation**

`nrows` is useful for a quick sample. A small sample may not reveal all data-quality problems.



### 8.3 usecols

Read only selected columns when the task requires them.

### 8.3. Read selected columns

```python
import pandas as pd

df = pd.read_csv(
    "data/customer_churn.csv",
    usecols=["customer_id", "age", "churn"]
)

print(df.head())
```

**Expected output**

```text
The selected columns from the file, if all names exist.
```

**Explanation**

`usecols` reduces the data loaded into memory. Confirm that selected column names match the source file.



### 8.4 index_col

Choose a column to use as the index.

### 8.4. Set customer_id as the index

```python
import pandas as pd

df = pd.read_csv(
    "data/customer_churn.csv",
    index_col="customer_id"
)

print(df.head())
```

**Expected output**

```text
A table where customer_id appears as the row index rather than a regular column.
```

**Explanation**

Use an index when it makes row identification convenient. Do not assume an identifier is unique without checking it.



### 8.5 encoding

Specify the file encoding when the source requires it.

### 8.5. Specify an encoding

```python
import pandas as pd

df = pd.read_csv(
    "data/customer_churn.csv",
    encoding="utf-8"
)

print(df.head())
```

**Expected output**

```text
The first five rows if the file is valid UTF-8.
```

**Explanation**

UTF-8 is common, but a file may use another encoding. If decoding fails, identify the source encoding rather than guessing repeatedly.



### 8.6 dtype

Set a column's type when the data meaning and source format are understood.

### 8.6. Read an identifier as text

```python
import pandas as pd

df = pd.read_csv(
    "data/customer_churn.csv",
    dtype={"customer_id": "string"}
)

print(df["customer_id"].dtype)
```

**Expected output**

```text
string
```

**Explanation**

An identifier such as C001 is text. Even numeric-looking identifiers may need to remain strings to preserve leading zeros and avoid arithmetic interpretation.



### 8.7 na_values

Specify additional markers that should be interpreted as missing.

### 8.7. Recognize a custom missing marker

```python
import pandas as pd

df = pd.read_csv(
    "data/customer_churn.csv",
    na_values=["unknown", "not_provided"]
)

print(df.head())
```

**Expected output**

```text
The first five rows, with the specified markers parsed as missing values.
```

**Explanation**

Only add markers that the data owner confirms mean missing information. A word such as 'unknown' may have a business meaning in some datasets.



## 9. Read Excel files

Pandas can read Excel workbooks when a compatible engine is installed.

### 9.1 Read an Excel workbook

### 9.1. Read an Excel file

```python
import pandas as pd

df = pd.read_excel("data/customer_churn.xlsx")

print(df.head())
```

**Expected output**

```text
The first five rows from the default worksheet, assuming the file and required engine are available.
```

**Explanation**

`read_excel()` reads Excel files. Depending on the file format and environment, Pandas may require an optional engine such as openpyxl.



### 9.2 Select a worksheet by name

### 9.2. Read a named worksheet

```python
import pandas as pd

df = pd.read_excel(
    "data/customer_churn.xlsx",
    sheet_name="Customers"
)

print(df.head())
```

**Expected output**

```text
The first five rows from the worksheet named Customers, if it exists.
```

**Explanation**

The sheet name must match the workbook. Inspect the workbook or ask the data owner if the sheet name is unknown.



### 9.3 Read a specific number of rows

### 9.3. Read a small Excel sample

```python
import pandas as pd

sample_df = pd.read_excel(
    "data/customer_churn.xlsx",
    nrows=10
)

print(sample_df.shape)
```

**Expected output**

```text
A table containing at most ten rows and the selected worksheet's columns.
```

**Explanation**

This can help inspect a workbook without immediately loading every row.



## 10. Inspect a new dataset

Before training a model, understand the dataset structure. Do not begin model training immediately after loading a file.

### 10.1 View the first five rows

### 10.1. Use head()

```python
print(df.head())
```

**Expected output**

```text
The first five rows of the DataFrame.
```

**Explanation**

`head()` is useful for a quick visual inspection. It does not validate the entire dataset.



### 10.2 View the first three rows

### 10.2. Pass a row count to head()

```python
print(df.head(3))
```

**Expected output**

```text
The first three rows.
```

**Explanation**

The integer argument controls how many rows are displayed.



### 10.3 View the last five rows

### 10.3. Use tail()

```python
print(df.tail())
```

**Expected output**

```text
The last five rows.
```

**Explanation**

The final rows may reveal trailing blank records or unusual entries, but a full quality check requires more than visual inspection.



### 10.4 Inspect column names

### 10.4. Print all column labels

```python
print(df.columns.tolist())
```

**Expected output**

```text
A Python list containing the DataFrame column names.
```

**Explanation**

`tolist()` converts the columns index into a regular Python list.



### 10.5 Inspect data types

### 10.5. Use dtypes

```python
print(df.dtypes)
```

**Expected output**

```text
A Series showing each column's data type.
```

**Explanation**

This helps identify numeric, text and Boolean-like columns. A numeric-looking field may still be stored as text.



### 10.6 Inspect non-null counts and types

### 10.6. Use info()

```python
df.info()
```

**Expected output**

```text
A summary containing the index range, column names, non-null counts, dtypes and memory usage.
```

**Explanation**

`info()` is one of the most useful first inspections. Non-null counts help identify columns with missing values.



### 10.7 Summary statistics for numeric columns

### 10.7. Use describe()

```python
print(df.describe())
```

**Expected output**

```text
A table of summary statistics for numeric columns, such as count, mean, standard deviation, minimum and quartiles.
```

**Explanation**

This is a descriptive summary. It does not explain business meaning or prove that the values are valid.



### 10.8 Include non-numeric columns in a summary

### 10.8. Describe all column types

```python
print(df.describe(include="all"))
```

**Expected output**

```text
A broader summary that may include counts, unique values, top values and frequencies for non-numeric columns.
```

**Explanation**

Different column types have different summary fields. Missing values can affect the counts.



## 11. Selecting columns

### 11.1 Select one column

### 11.1. Select a Series by column name

```python
age_series = df["age"]

print(age_series.head())
```

**Expected output**

```text
The first five values of the age column.
```

**Explanation**

Single square-bracket selection usually returns a Series.



### 11.2 Select multiple columns

### 11.2. Select multiple columns

```python
selected_df = df[["customer_id", "age", "churn"]]

print(selected_df.head())
```

**Expected output**

```text
A DataFrame containing the selected columns.
```

**Explanation**

The inner list contains the column names. The outer brackets select from the DataFrame.



### 11.3 Series versus DataFrame

### 11.3. Compare single and double brackets

```python
single_column = df["age"]
one_column_df = df[["age"]]

print(type(single_column))
print(type(one_column_df))
```

**Expected output**

```text
<class 'pandas.core.series.Series'>
<class 'pandas.core.frame.DataFrame'>
```

**Explanation**

One pair of brackets returns a Series; a list of column names returns a DataFrame, even when the list contains one column.



## 12. Selecting rows with loc and iloc

### 12.1 loc selects by label

### 12.1. Select a row by index label

```python
print(df.loc[0])
```

**Expected output**

```text
The row whose index label is 0, if that label exists.
```

**Explanation**

`loc` uses index labels. The label may not be a simple integer after the DataFrame index has been changed.



### 12.2 iloc selects by integer position

### 12.2. Select the first row by position

```python
print(df.iloc[0])
```

**Expected output**

```text
The first row as a Series.
```

**Explanation**

`iloc` uses zero-based integer position, regardless of the row label.



### 12.3 Select rows and columns with loc

### 12.3. Select a labelled row and columns

```python
print(df.loc[0, ["age", "churn"]])
```

**Expected output**

```text
The age and churn values for the row labelled 0, if it exists.
```

**Explanation**

The first argument is the row label and the second argument selects the requested columns.



### 12.4 Select rows and columns with iloc

### 12.4. Select by row and column position

```python
print(df.iloc[0:3, 0:2])
```

**Expected output**

```text
The first three rows and first two columns.
```

**Explanation**

The stop position is excluded, as with ordinary Python slicing.



## 13. Filtering rows

Filtering selects records that satisfy a condition.

### 13.1 Filter one condition

### 13.1. Select customers older than 30

```python
older_customers = df[df["age"] > 30]

print(older_customers.head())
```

**Expected output**

```text
Rows where age is greater than 30.
```

**Explanation**

The expression creates a Boolean mask and uses it to select rows.



### 13.2 Filter equality

### 13.2. Select monthly contracts

```python
monthly_customers = df[df["contract_type"] == "Monthly"]

print(monthly_customers.head())
```

**Expected output**

```text
Rows where contract_type equals Monthly, assuming that column exists.
```

**Explanation**

Use `==` for comparison. Text comparisons depend on exact values, including capitalization and whitespace.



### 13.3 Filter with AND

### 13.3. Combine two conditions with &

```python
selected = df[
    (df["age"] > 30) & (df["monthly_charges"] > 500)
]

print(selected.head())
```

**Expected output**

```text
Rows where both conditions are true, assuming the columns exist and are numeric.
```

**Explanation**

Use `&` for element-wise AND in Pandas Boolean masks. Put each condition inside parentheses.



### 13.4 Filter with OR

### 13.4. Combine conditions with |

```python
selected = df[
    (df["contract_type"] == "Monthly") |
    (df["contract_type"] == "Annual")
]

print(selected.head())
```

**Expected output**

```text
Rows matching either contract type, assuming the column exists.
```

**Explanation**

Use `|` for element-wise OR. Parentheses prevent operator-precedence mistakes.



### 13.5 Filter using isin()

### 13.5. Select from a list of categories

```python
selected = df[df["contract_type"].isin(["Monthly", "Annual"])]

print(selected.head())
```

**Expected output**

```text
Rows whose contract_type value is in the supplied list.
```

**Explanation**

`isin()` is convenient when checking membership in several allowed categories.



## 14. Sorting data

### 14.1 Sort ascending

### 14.1. Sort by age

```python
sorted_df = df.sort_values(by="age")

print(sorted_df.head())
```

**Expected output**

```text
Rows ordered by age from smallest to largest.
```

**Explanation**

`sort_values()` returns a sorted DataFrame by default without changing the original variable unless assigned back.



### 14.2 Sort descending

### 14.2. Sort by charges descending

```python
sorted_df = df.sort_values(
    by="monthly_charges",
    ascending=False
)

print(sorted_df.head())
```

**Expected output**

```text
Rows with the highest monthly_charges first.
```

**Explanation**

`ascending=False` reverses the sort order.



### 14.3 Sort by multiple columns

### 14.3. Sort by contract and age

```python
sorted_df = df.sort_values(
    by=["contract_type", "age"],
    ascending=[True, False]
)

print(sorted_df.head())
```

**Expected output**

```text
Rows sorted by contract_type ascending and age descending within each contract group.
```

**Explanation**

The `ascending` list corresponds to the order of the `by` columns.



## 15. Creating and modifying columns

### 15.1 Create a simple derived column

### 15.1. Create a yearly charge column

```python
df["estimated_yearly_charges"] = df["monthly_charges"] * 12

print(df[["monthly_charges", "estimated_yearly_charges"]].head())
```

**Expected output**

```text
A two-column view showing monthly charges and the derived yearly estimate.
```

**Explanation**

The new column is calculated row by row. Confirm that monthly_charges is numeric and that multiplying by twelve is meaningful for the business definition.



### 15.2 Create a Boolean flag

### 15.2. Create a high-charge flag

```python
df["high_charge"] = df["monthly_charges"] > 500

print(df[["monthly_charges", "high_charge"]].head())
```

**Expected output**

```text
A Boolean column indicating whether each value is above 500.
```

**Explanation**

The comparison returns one Boolean value per row.



### 15.3 Rename columns

### 15.3. Rename a column

```python
df = df.rename(columns={"monthly_charges": "monthly_fee"})

print(df.columns.tolist())
```

**Expected output**

```text
The column list with monthly_fee instead of monthly_charges.
```

**Explanation**

`rename()` returns a DataFrame with the renamed column. Assign the result if you want to keep the change.



### 15.4 Drop a column

### 15.4. Create a DataFrame without an identifier column

```python
model_input = df.drop(columns=["customer_id"])

print(model_input.head())
```

**Expected output**

```text
A DataFrame without customer_id, assuming the column exists.
```

**Explanation**

`drop()` returns a new object by default. An identifier may be excluded from model inputs if it is not a valid predictive feature, but that decision depends on the task.



## 16. Missing-value awareness

This lesson introduces how to inspect missing values. Full cleaning and treatment are covered on Day 10.

### 16.1 Check missing values by column

### 16.1. Count missing values

```python
print(df.isnull().sum())
```

**Expected output**

```text
A count of missing values for each column.
```

**Explanation**

`isnull()` creates a Boolean DataFrame, and `sum()` counts true values column by column.



### 16.2 Check whether any value is missing

### 16.2. Check the whole DataFrame

```python
print(df.isnull().values.any())
```

**Expected output**

```text
True if at least one missing value exists; otherwise False.
```

**Explanation**

This gives a quick overall check. It does not show which columns or rows contain missing values.



### 16.3 Calculate missing-value percentages

### 16.3. Calculate missing percentage by column

```python
missing_percent = df.isnull().mean() * 100

print(missing_percent)
```

**Expected output**

```text
A percentage for each column.
```

**Explanation**

The mean of a Boolean missing-value indicator gives the missing fraction. Multiplying by 100 converts it to a percentage.



## 17. Duplicate awareness

Full duplicate detection and treatment are covered on Day 10.

### 17.1 Check duplicate rows

### 17.1. Count duplicate rows

```python
print(df.duplicated().sum())
```

**Expected output**

```text
The number of rows that are duplicates of earlier rows, using all columns by default.
```

**Explanation**

This is a quick inspection. Whether a row is a true duplicate depends on the business key and data meaning.



### 17.2 Check duplicates using an identifier

### 17.2. Check repeated customer IDs

```python
print(df["customer_id"].duplicated().sum())
```

**Expected output**

```text
The number of repeated customer_id values after the first occurrence, assuming the column exists.
```

**Explanation**

Use this only if customer_id is expected to identify one record per customer. Some datasets legitimately contain multiple events per customer.



## 18. Basic group summaries

### 18.1 Count categories

### 18.1. Count contract types

```python
print(df["contract_type"].value_counts())
```

**Expected output**

```text
A frequency count for each contract_type category.
```

**Explanation**

`value_counts()` helps inspect categorical distribution. By default, missing values are excluded.



### 18.2 Include missing values in category counts

### 18.2. Count categories including missing

```python
print(df["contract_type"].value_counts(dropna=False))
```

**Expected output**

```text
A frequency count that includes a missing-value category when present.
```

**Explanation**

The `dropna=False` argument includes missing values in the count.



### 18.3 Group and calculate an average

### 18.3. Average charges by contract type

```python
summary = df.groupby("contract_type")["monthly_charges"].mean()

print(summary)
```

**Expected output**

```text
An average monthly_charges value for each contract type, assuming the columns exist and are numeric.
```

**Explanation**

`groupby()` creates groups using the category, and `mean()` calculates the mean within each group.



## 19. Save a DataFrame

### 19.1 Save to CSV

### 19.1. Export selected data

```python
df.to_csv("data/customer_churn_inspected.csv", index=False)

print("CSV file saved")
```

**Expected output**

```text
CSV file saved
```

**Explanation**

The DataFrame is written to the supplied path. `index=False` prevents Pandas from writing the DataFrame index as an extra column.



### 19.2 Save to Excel

### 19.2. Export to Excel

```python
df.to_excel("data/customer_churn_inspected.xlsx", index=False)

print("Excel file saved")
```

**Expected output**

```text
Excel file saved
```

**Explanation**

Excel writing may require an optional engine such as openpyxl. Confirm the destination directory exists and the environment has the required package.



## 20. A step-by-step inspection workflow

For a new ML dataset, a sensible first inspection sequence is:

1. Confirm the file path and format.
2. Read the file.
3. Inspect a few rows.
4. Check the shape and column names.
5. Check data types and non-null counts.
6. Inspect summary statistics.
7. Check category frequencies.
8. Check missing values and duplicates.
9. Confirm the target column and observation unit.
10. Record issues for the data-cleaning stage.

This is an inspection workflow, not a complete preprocessing pipeline. Do not silently clean data before understanding its meaning.

## 21. Common beginner mistakes

| Mistake | Why it happens | Better approach |
|---|---|---|
| Wrong file path | Relative path is based on current working directory | Inspect `os.getcwd()` and verify the path |
| Wrong delimiter | Source file is not comma-separated | Confirm the delimiter and set `sep` |
| Wrong encoding | File uses a different character encoding | Identify the source encoding |
| Wrong data type | Numeric-looking values are stored as text | Inspect `dtypes` and actual values |
| Assuming head() validates everything | It shows only a few rows | Use structural and quality checks |
| Confusing loc and iloc | Label and position selection differ | Use loc for labels and iloc for positions |
| Using and/or with Boolean Series | Python scalar logical operators are not vectorized masks | Use `&` and `|` with parentheses |
| Overwriting the source file | Output path matches the original | Save cleaned or inspected output separately |
| Treating every ID as a feature | An ID may be a key, not a meaningful predictor | Understand its business meaning before model use |

## 22. ML context: features and target

A DataFrame makes it easy to separate the input columns from the target after the dataset has been inspected.

### 22.1 Select feature columns

```python
feature_columns = ["age", "monthly_charges", "contract_type"]

X = df[feature_columns]

print(X.head())
```

**Expected output**

The first rows of the selected feature columns, assuming they exist in the DataFrame.

**Explanation:** `X` is a conventional variable name for the feature table. Feature preparation such as encoding and scaling is covered later.

### 22.2 Select the target column

```python
y = df["churn"]

print(y.head())
```

**Expected output**

The first values from the churn column, assuming it exists.

**Explanation:** `y` is a common name for the target values in supervised learning. Confirm that the target is correctly defined and aligned with the feature rows.

### 22.3 Check row counts

```python
print(X.shape[0])
print(y.shape[0])
print(X.shape[0] == y.shape[0])
```

**Expected output**

The feature row count, the target row count and True if they match.

**Explanation:** Features and target values must correspond to the same observations. Equal row counts are necessary but not sufficient; row order and index alignment also matter.

## 23. Interview-ready explanations

**What is the difference between a Series and a DataFrame?**  
A Series is a one-dimensional labelled collection. A DataFrame is a two-dimensional labelled table with rows and columns. A DataFrame can contain columns with different data types.

**How do you read a CSV file?**  
Use `pd.read_csv(path)`. Confirm the file path, delimiter and encoding. Inspect the resulting DataFrame instead of assuming the file loaded as intended.

**What is the difference between loc and iloc?**  
`loc` selects by index labels. `iloc` selects by integer position.

**Why inspect dtypes and info() before ML?**  
They help identify unexpected types, missing values and structural problems. A numeric feature stored as text may fail during numerical preprocessing.

**Does head() prove that a dataset is clean?**  
No. It displays only a small number of rows. Use structural summaries, missing-value checks, duplicate checks and domain-specific validation.

## 24. Lesson completion checklist

- [ ] I can import Pandas.
- [ ] I can create a Series and DataFrame.
- [ ] I understand index labels, rows and columns.
- [ ] I can read a CSV file using a valid path.
- [ ] I know when to use read_csv() parameters.
- [ ] I can read an Excel workbook when the engine is available.
- [ ] I can inspect shape, columns, dtypes, info() and describe().
- [ ] I can select columns and rows with loc and iloc.
- [ ] I can filter and sort records.
- [ ] I can create a simple derived column.
- [ ] I can inspect missing values and duplicates.
- [ ] I can separate feature columns and a target column.

**Next lesson:** Day 07 — Linear Algebra: Scalars, Vectors and Matrices. We will connect the shape of numerical data to the way machine-learning algorithms represent observations and features.
