# Day 06: Pandas Series, DataFrames and File Reading

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate  
**Learning focus:** Understand the concepts, apply them correctly, and explain decisions in an interview.

---

## 1. Learning Objectives

- Explain the roles of Pandas Series and DataFrames.
- Create DataFrames from dictionaries and lists.
- Read CSV and Excel files.
- Inspect rows, columns, types and summary information.
- Select, filter, sort and create columns.
- Understand missing values and basic grouping.
- Recognize common file-reading and schema problems.

## 2. What Is Pandas?

Pandas is a Python library for working with labelled, tabular and time-series data. It is widely used for loading, inspecting, cleaning and preparing datasets before machine learning.

```python
import pandas as pd
```

## 3. Series and DataFrame

### Series

A Series is a one-dimensional labelled array.

```python
import pandas as pd

ages = pd.Series([25, 31, 40], name="age")
print(ages)
```

A Series contains values and an index. It is often used to represent one column.

### DataFrame

A DataFrame is a two-dimensional table with rows and columns.

```python
customers = pd.DataFrame({
    "customer": ["Asha", "Ravi", "John"],
    "age": [25, 31, 40],
    "monthly_spend": [1200, 2500, 1800]
})

print(customers)
```

Output:

```text
  customer  age  monthly_spend
0     Asha   25           1200
1     Ravi   31           2500
2     John   40           1800
```

Each row is an observation; each column is a variable/feature in this simple example.

## 4. Read Data from Files

### Read a CSV file

```python
import pandas as pd

df = pd.read_csv("customers.csv")
print(df.head())
```

`read_csv()` loads comma-separated values into a DataFrame. The path must point to a file available to the Python process.

Useful parameters:

| Parameter | Purpose |
|---|---|
| `filepath_or_buffer` | File path or readable source |
| `sep` | Delimiter, such as `,` or `;` |
| `encoding` | Text encoding, such as `utf-8` |
| ` na_values` / `na_values` | Extra strings to interpret as missing |
| `usecols` | Read only selected columns |
| `parse_dates` | Parse selected date columns |
| `dtype` | Specify selected column types when appropriate |

Example:

```python
df = pd.read_csv(
    "customers.csv",
    usecols=["customer_id", "age", "monthly_spend"],
    na_values=["NA", "unknown", "-"]
)
```

### Read an Excel file

```python
df = pd.read_excel("sales.xlsx", sheet_name="Orders")
print(df.head())
```

Excel reading may require an engine package such as `openpyxl` for `.xlsx` files:

```bash
python -m pip install openpyxl
```

## 5. Inspect a Dataset Before Analysis

For ML work, never assume the dataset has the expected structure.

```python
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
```

| Method / attribute | What it tells you |
|---|---|
| `head()` | First five rows by default |
| `tail()` | Last five rows by default |
| `shape` | Tuple `(rows, columns)` |
| `columns` | Column labels |
| `dtypes` | Data type of each column |
| `info()` | Non-null counts, dtypes and memory summary |
| `describe()` | Numeric summary statistics by default |
| `isna().sum()` | Missing count in each column |
| `nunique()` | Number of unique values per column |

### Numeric and categorical summaries

```python
print(df.describe())
print(df.describe(include="all"))
```

`describe()` is a quick summary, not a data-quality certificate. Check units, valid ranges and business meaning separately.

## 6. Select Columns and Rows

### Select one or multiple columns

```python
age_series = df["age"]
age_table = df[["customer_id", "age"]]
```

One set of square brackets usually returns a Series for one column; a list of column names returns a DataFrame.

### `loc` and `iloc`

- `loc` selects by label.
- `iloc` selects by integer position.

```python
print(df.loc[0, "age"])
print(df.iloc[0, 1])
print(df.iloc[:5, :3])
```

The exact result depends on the DataFrame's index and column order.

## 7. Filter Rows

Boolean filtering selects records that match a condition.

```python
high_value = df[df["monthly_spend"] >= 2000]
print(high_value)
```

Multiple conditions:

```python
selected = df[
    (df["age"] >= 25) &
    (df["monthly_spend"] > 1500)
]
```

Use `&` for element-wise AND and `|` for element-wise OR; put each comparison in parentheses. Do not use Python's `and` for two whole pandas Series.

## 8. Create and Modify Columns

```python
df["annual_spend_estimate"] = df["monthly_spend"] * 12
```

This creates a new column calculated from an existing column.

A feature created this way may be valid for a model only if the underlying information is available at the prediction time and does not leak future information.

## 9. Sort and Group Data

### Sort

```python
sorted_df = df.sort_values("monthly_spend", ascending=False)
```

### Group and aggregate

```python
region_summary = df.groupby("region")["monthly_spend"].mean()
print(region_summary)
```

This calculates the average monthly spend per region, assuming a `region` column exists and values are meaningful.

For multiple aggregations:

```python
summary = df.groupby("region").agg(
    customer_count=("customer_id", "nunique"),
    average_spend=("monthly_spend", "mean"),
    total_spend=("monthly_spend", "sum")
)
```

## 10. Missing Values — First Inspection

```python
print(df.isna().sum())
print(df.isna().mean() * 100)
```

The first line counts missing cells per column. The second estimates missing percentage per column.

Do not immediately drop or fill every missing value. First ask:

- Does missing mean unknown, not applicable or not recorded?
- Is the missingness concentrated in a group or time period?
- Is the column required for the task?
- Would filling it introduce bias or distort the meaning?

Detailed treatment is covered in the Data Preparation phase.

## 11. Industry Scenario — E-Commerce Orders

The company wants to compare average order value by region.

```python
import pandas as pd

orders = pd.DataFrame({
    "order_id": ["O1", "O2", "O3", "O4", "O5"],
    "region": ["South", "North", "South", "West", "North"],
    "amount": [1200, 1800, 800, 2200, 1400]
})

print(orders.groupby("region")["amount"].mean())
```

Output:

```text
region
North    1600.0
South    1000.0
West     2200.0
Name: amount, dtype: float64
```

Interpretation: In this small example, West has the highest average order amount. This alone does not prove West is the most profitable region; costs, order counts and sample size also matter.

## 12. Common Pandas Errors

| Problem | Likely reason | First check |
|---|---|---|
| `FileNotFoundError` | Incorrect path or working directory | `Path.cwd()` and file path |
| `KeyError: 'age'` | Column name absent or differently spelled | `df.columns.tolist()` |
| `ParserError` | Inconsistent delimiter/quoting or malformed file | Inspect source and `sep` |
| Numeric column read as `object` | Text, symbols or mixed values | `df['col'].head()` and `to_numeric` validation |
| Filter returns no rows | Condition or value mismatch | Inspect unique values and ranges |
| Excel import engine error | Required package missing | Install the correct Excel engine |

### Convert a column carefully

```python
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
```

`errors="coerce"` turns invalid conversions into missing values. Always count and investigate those new missing entries; do not silently ignore them.

## 13. Quick Revision

- Series is one-dimensional; DataFrame is two-dimensional.
- Use `read_csv()` and `read_excel()` to load common business files.
- Inspect shape, columns, dtypes, null counts and summaries before analysis.
- Use column selection, `loc`, `iloc`, Boolean filters, sorting and `groupby` for common analysis.
- Missing values and type conversions require investigation, not blind fixes.