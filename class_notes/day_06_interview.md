# Day 06: Interview Questions — Pandas Series, DataFrames and File Reading

**Pathway:** Machine Learning Engineer

---

## Question 1: What is the difference between a Pandas Series and a DataFrame?

**Answer:** A Series is a one-dimensional labelled array. A DataFrame is a two-dimensional labelled table made of columns that can have different data types.

**Follow-up:** What does `df["age"]` usually return?

**Answer:** A Series. `df[["age"]]` returns a DataFrame with one column.

## Question 2: How do you read a CSV file in Pandas?

**Answer:** Use `pd.read_csv("file.csv")`. You can configure delimiter, encoding, selected columns, missing-value markers and data types using parameters.

**Follow-up:** What would you do if the file uses semicolons?

**Answer:** Use `pd.read_csv("file.csv", sep=";")` after confirming the delimiter.

## Question 3: How do you inspect a new dataset before model training?

**Answer:** I inspect `head()`, `shape`, `columns`, `dtypes`, `info()`, `describe()`, missing-value counts, unique values and the target distribution. I also validate the schema and business meaning.

**Follow-up:** Does `describe()` identify every data problem?

**Answer:** No. It provides summary statistics but cannot determine all invalid values, leakage, duplicate business events or incorrect units.

## Question 4: What is the difference between `loc` and `iloc`?

**Answer:** `loc` selects by index and column labels; `iloc` selects by integer positions.

**Follow-up:** Does `iloc[0]` select the first row?

**Answer:** Yes, positional row 0, irrespective of the actual index label.

## Question 5: How do you filter rows where age is above 30 and income is above 50,000?

**Answer:**

```python
df[(df["age"] > 30) & (df["income"] > 50000)]
```

Each comparison is parenthesized and `&` combines element-wise conditions.

**Follow-up:** Why not write `and`?

**Answer:** Pandas Series contain many Boolean values. Python's scalar `and` expects a single truth value; `&` performs element-wise combination.

## Question 6: Your numerical column is loaded as `object`. What do you do?

**Answer:** Inspect its values for currency symbols, commas, text markers or mixed entries. Clean and validate the representation, then convert using `pd.to_numeric`. If using `errors="coerce"`, count and investigate newly missing values.

**Follow-up:** Should you remove all invalid rows immediately?

**Answer:** No. Determine why the values are invalid, how many records are affected and whether removal would bias the dataset.

## Question 7: What does `groupby()` do?

**Answer:** It divides records into groups based on one or more keys and allows aggregation within each group, such as mean, sum, count or minimum/maximum.

**Example:** Calculate average sales per region.

**Follow-up:** Why include customer count along with average order value?

**Answer:** An average based on a very small number of records may be unstable or misleading. Counts provide context.

## Question 8: A CSV file says it contains 10,000 rows, but Pandas loads 9,800. What would you investigate?

**Answer:** Check whether the source count includes a header, blank lines or malformed records; inspect parser warnings, delimiter and quoting; verify file version and filters; and compare row counts after each processing step. I would not silently assume the missing 200 rows are irrelevant.

**Follow-up:** Why is row-count reconciliation important?

**Answer:** It helps detect data loss or unintended filtering before analysis and model training.