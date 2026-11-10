# Day 06: Pandas Series, DataFrames and File Reading

**Pathway:** Machine Learning Engineer  
**Phase:** 1 — Python, mathematics and statistics foundations  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define Pandas, Series and DataFrame.
- Create Series and DataFrames.
- Read CSV and Excel files.
- Inspect rows, columns, data types and summary statistics.
- Select, filter, sort and create columns.
- Understand common file-reading problems.

## 1. What Is Pandas?
Pandas is a Python library used to work with labelled tabular data. It is widely used for data cleaning, analysis and preparing data for machine learning.

Import convention:
```python
import pandas as pd
```

## 2. What Is a Series?
A **Series** is a one-dimensional labelled collection of values. It has an index and values.

**Syntax:** `pd.Series(data, index=...)`

**Example 1**
```python
marks = pd.Series([80, 90, 75])
print(marks)
```
Output shows index 0, 1, 2 and corresponding values.

**Example 2 — Custom index**
```python
sales = pd.Series([5000, 7000], index=["Jan", "Feb"])
print(sales["Feb"])
```
Output: 7000.

**Example 3 — Dictionary to Series**
```python
scores = pd.Series({"Asha": 85, "Ravi": 90})
print(scores["Asha"])
```
Output: 85.

## 3. What Is a DataFrame?
A **DataFrame** is a two-dimensional labelled table with rows and columns. Columns can contain different data types.

**Example 1 — Create from dictionary**
```python
df = pd.DataFrame({
    "name": ["Asha", "Ravi"],
    "age": [25, 30],
    "city": ["Hyd", "Vizag"]
})
print(df)
```

**Example 2 — Select columns**
```python
print(df["name"])
print(df[["name", "city"]])
```
A single column selection returns a Series; selecting a list of columns returns a DataFrame.

**Example 3 — Select rows**
```python
print(df.loc[0, "name"])
print(df.iloc[1, 1])
```
`loc` selects using labels; `iloc` selects using integer positions.

## 4. Reading Files
**CSV syntax:** `pd.read_csv("file.csv")`  
**Excel syntax:** `pd.read_excel("file.xlsx")`

**Example 1**
```python
df = pd.read_csv("customers.csv")
print(df.head())
```

**Example 2**
```python
df = pd.read_excel("sales.xlsx", sheet_name="January")
print(df.shape)
```

**Example 3 — Read selected columns**
```python
df = pd.read_csv("orders.csv", usecols=["order_id", "amount"])
print(df.head())
```

File paths must point to an existing accessible file. Excel reading may require an installed engine such as openpyxl.

## 5. Inspecting a DataFrame
| Method/property | Use |
|---|---|
| df.head() | First rows |
| df.tail() | Last rows |
| df.shape | Rows and columns |
| df.columns | Column names |
| df.info() | Data types and non-null counts |
| df.dtypes | Type of each column |
| df.describe() | Numeric summary |
| df.isna().sum() | Missing count per column |

**Examples**
```python
print(df.shape)
print(df.columns.tolist())
```

```python
df.info()
```

```python
print(df.describe(include="all"))
```

## 6. Filtering, Sorting and Creating Columns
**Example 1 — Filter**
```python
high_value = df[df["amount"] > 1000]
print(high_value)
```

**Example 2 — Sort**
```python
sorted_df = df.sort_values("amount", ascending=False)
print(sorted_df.head())
```

**Example 3 — Create derived column**
```python
df["total"] = df["price"] * df["quantity"]
print(df[["price", "quantity", "total"]])
```

## 7. Why DataFrame Inspection Matters for ML
A numeric-looking column may be stored as text; a date may not yet be a datetime; a column may contain many missing values. Inspect the schema before model training.

**Scenario questions**
1. A date column sorts alphabetically. What should you inspect? Its dtype and whether it should be converted to datetime.
2. Why does df.shape matter? It confirms dataset size and dimensions.
3. Why use loc vs iloc? loc selects by labels; iloc selects by row/column positions.
