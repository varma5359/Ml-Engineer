# Day 14: EDA, Distributions and Feature Relationships

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain Exploratory Data Analysis (EDA).
- Follow a structured EDA workflow before modelling.
- Inspect a dataset's structure, types and quality.
- Understand numeric and categorical feature distributions.
- Compare groups using Pandas summaries.
- Study relationships between features and a target.
- Choose a chart that answers a specific question.
- Recognize skewness, unusual values, imbalance and possible data leakage.
- Convert EDA findings into clear observations without claiming causation.

## 2. What is Exploratory Data Analysis?

**Exploratory Data Analysis (EDA)** is the process of examining a dataset to understand its structure, quality, distributions, patterns and relationships before building a model.

EDA helps answer questions such as:
- How many observations and features are available?
- Which columns contain missing values?
- Are categories balanced or rare?
- Are numerical values skewed?
- Are there unusual observations?
- Which features appear related to the target?
- Are there duplicate or redundant fields?
- Is the dataset suitable for the intended prediction task?

EDA is not a collection of random charts. Every inspection should answer a question.

## 3. Create an example customer dataset

We will use a small subscription-business dataset. Each row represents one customer at a defined snapshot.

### 3.1 Import libraries

### 3.1. Import Pandas and NumPy

```python
import pandas as pd
import numpy as np

print("Data libraries ready")
```

**Expected output**

```text
Data libraries ready
```

**Explanation**

Pandas manages the table and NumPy supports numerical operations.



### 3.2 Create the sample records

### 3.2. Create the customer dataset

```python
df = pd.DataFrame({
    "customer_id": [f"C{i:03d}" for i in range(1, 16)],
    "age": [22, 25, 31, 29, 42, 35, 28, 50, 45, 33, 27, 39, 41, 24, 36],
    "monthly_spend": [1200, 1800, 4500, 3200, 7000, 5200, 2100, 9000, 7500, 4800, 1900, 6100, 6800, 1500, 5500],
    "tenure_months": [2, 5, 18, 12, 36, 24, 6, 48, 42, 20, 4, 30, 34, 3, 26],
    "support_calls": [3, 2, 1, 2, 0, 1, 4, 0, 1, 2, 3, 1, 0, 4, 1],
    "city": ["Hyderabad", "Chennai", "Hyderabad", "Vizag", "Chennai", "Vizag", "Hyderabad", "Chennai", "Vizag", "Hyderabad", "Chennai", "Vizag", "Hyderabad", "Chennai", "Vizag"],
    "contract_type": ["Monthly", "Monthly", "Annual", "Annual", "Annual", "Annual", "Monthly", "Annual", "Annual", "Monthly", "Monthly", "Annual", "Annual", "Monthly", "Annual"],
    "churn": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
})

print(df.head())
```

**Expected output**

```text
The first five customer records
```

**Explanation**

The dataset contains numerical and categorical features plus a binary churn target. It is a teaching sample, not a statistically representative business dataset.



### 3.3 Inspect the number of rows and columns

### 3.3. Check dataset shape

```python
print(df.shape)
```

**Expected output**

```text
(15, 8)
```

**Explanation**

The dataset has 15 rows and eight columns.



### 3.4 Inspect column names

### 3.4. List column names

```python
print(df.columns.tolist())
```

**Expected output**

```text
A list of eight column names
```

**Explanation**

Confirm that the expected fields are present before writing analysis code.



### 3.5 Inspect data types and non-null counts

### 3.5. Use info()

```python
df.info()
```

**Expected output**

```text
A summary of column names, non-null counts, data types and memory use
```

**Explanation**

This provides a structural overview. It does not identify every invalid value.



### 3.6 Inspect first and last records

### 3.6. Use head() and tail()

```python
print(df.head(3))
print(df.tail(3))
```

**Expected output**

```text
The first three and final three rows
```

**Explanation**

Looking at both ends can reveal unusual records or ordering issues. It is still only a sample of the data.



## 4. Understand the observation unit and target

Before interpreting patterns, confirm what one row represents and what the target means.

In this teaching dataset:
- One row represents one customer at a snapshot.
- `churn` is the target label.
- `customer_id` is an identifier.
- Other columns are candidate features.

### 4.1 Confirm unique customer IDs

### 4.1. Check customer ID uniqueness

```python
print(df["customer_id"].is_unique)
```

**Expected output**

```text
True
```

**Explanation**

The IDs are unique in this sample. A production dataset may contain repeated entities or a different row grain.



### 4.2 Inspect target counts

### 4.2. Count churn labels

```python
print(df["churn"].value_counts())
```

**Expected output**

```text
Counts for churn 0 and churn 1
```

**Explanation**

This shows the number of rows in each target class.



### 4.3 Inspect target proportions

### 4.3. Calculate target proportions

```python
print(df["churn"].value_counts(normalize=True).sort_index())
```

**Expected output**

```text
The fraction of rows in each class
```

**Explanation**

Class proportions help identify imbalance. This small sample should not be used to make broad claims about the business population.



### 4.4 Inspect the target meaning

Write down:
- What event does churn=1 represent?
- What does churn=0 represent?
- When is the label assigned?
- What future period does the label cover?
- Which information is available at prediction time?

Without a clear target definition, an EDA chart may be technically correct but answer the wrong question.

## 5. Numeric feature distributions

A **distribution** describes how values occur across observations. It helps us understand typical values, spread, skewness, gaps and unusual observations.

### 5.1 Select numeric columns

### 5.1. Select numerical features

```python
numeric_df = df.select_dtypes(include="number")

print(numeric_df.columns.tolist())
```

**Expected output**

```text
A list containing age, monthly_spend, tenure_months, support_calls and churn
```

**Explanation**

The target is numeric too, so keep the target separate when analysing input feature distributions.



### 5.2 Describe numeric columns

### 5.2. Use describe()

```python
print(numeric_df.describe().T)
```

**Expected output**

```text
A summary table with count, mean, standard deviation, minimum, quartiles and maximum for numeric columns
```

**Explanation**

The transpose makes each feature one row. Summary statistics are an initial overview, not a complete explanation of the distribution.



### 5.3 Inspect the age range

### 5.3. Check minimum and maximum age

```python
print(df["age"].min())
print(df["age"].max())
```

**Expected output**

```text
22 followed by 50
```

**Explanation**

The range gives the smallest and largest recorded ages. Confirm plausible business ranges before flagging values.



### 5.4 Inspect distinct values

### 5.4. Count unique values per feature

```python
print(df.nunique())
```

**Expected output**

```text
The number of distinct values in each column
```

**Explanation**

A feature with very few distinct values may be categorical or binary even if stored as a number.



### 5.5 Inspect skewness

### 5.5. Calculate feature skewness

```python
print(df[["age", "monthly_spend", "tenure_months", "support_calls"]].skew())
```

**Expected output**

```text
A skewness value for each numerical feature
```

**Explanation**

Skewness summarizes asymmetry. Interpret it together with a histogram and business context; small samples can produce unstable values.



## 6. Visualize numerical distributions

### 6.1 Import Matplotlib and Seaborn

### 6.1. Import plotting libraries

```python
import matplotlib.pyplot as plt
import seaborn as sns

print("Plotting libraries ready")
```

**Expected output**

```text
Plotting libraries ready
```

**Explanation**

Matplotlib provides core plotting functions. Seaborn offers convenient statistical charts built on Matplotlib.



### 6.2 Plot age distribution

### 6.2. Create an age histogram

```python
sns.histplot(data=df, x="age", bins=8)
plt.xlabel("Customer age (years)")
plt.ylabel("Number of customers")
plt.title("Customer age distribution")
plt.show()
```

**Expected output**

```text
A histogram showing how customer ages are distributed
```

**Explanation**

A histogram groups numerical values into bins. Choose a bin count that reveals patterns without hiding detail.



### 6.3 Plot monthly spend distribution

### 6.3. Create a spend histogram

```python
sns.histplot(data=df, x="monthly_spend", bins=8)
plt.xlabel("Monthly spend")
plt.ylabel("Number of customers")
plt.title("Monthly spend distribution")
plt.show()
```

**Expected output**

```text
A histogram of monthly spend
```

**Explanation**

Use the chart to inspect spread and skewness. The chart does not tell you whether high spend is good or bad for churn.



### 6.4 Compare histogram and KDE

### 6.4. Add a density curve

```python
sns.histplot(
    data=df,
    x="monthly_spend",
    bins=8,
    kde=True
)
plt.title("Monthly spend with density curve")
plt.show()
```

**Expected output**

```text
A histogram with a smoothed density estimate
```

**Explanation**

The density curve is an estimate that depends on smoothing. It should not be treated as an exact probability model, especially for small samples.



### 6.5 Plot tenure distribution

### 6.5. Inspect customer tenure

```python
sns.histplot(data=df, x="tenure_months", bins=8)
plt.xlabel("Tenure (months)")
plt.ylabel("Customers")
plt.title("Customer tenure distribution")
plt.show()
```

**Expected output**

```text
A histogram of customer tenure
```

**Explanation**

Tenure can be strongly affected by the customer acquisition period and data snapshot date. Interpret it with the collection window.



### 6.6 Plot support-call counts

### 6.6. Inspect support calls

```python
sns.histplot(data=df, x="support_calls", discrete=True)
plt.xlabel("Support calls")
plt.ylabel("Customers")
plt.title("Support calls per customer")
plt.show()
```

**Expected output**

```text
A discrete histogram of support call counts
```

**Explanation**

Counts are discrete values. The discrete option aligns bins with count values.



## 7. Categorical feature distributions

Categorical distributions show how often each label occurs.

### 7.1 Inspect city counts

### 7.1. Count customers by city

```python
print(df["city"].value_counts())
```

**Expected output**

```text
Counts for Hyderabad, Chennai and Vizag
```

**Explanation**

This reports row counts for each city. If one customer can appear multiple times, the result would count records rather than unique customers.



### 7.2 Plot city counts

### 7.2. Create a count plot

```python
sns.countplot(data=df, x="city")
plt.xlabel("City")
plt.ylabel("Number of customers")
plt.title("Customers by city")
plt.xticks(rotation=20)
plt.show()
```

**Expected output**

```text
A bar chart comparing counts across cities
```

**Explanation**

Count plots are useful for categorical frequencies. Rotate labels if they overlap.



### 7.3 Count contract types

### 7.3. Count contract categories

```python
print(df["contract_type"].value_counts())
```

**Expected output**

```text
Counts for Annual and Monthly contracts
```

**Explanation**

The counts show how many records belong to each contract group.



### 7.4 Plot contract categories

### 7.4. Visualize contract distribution

```python
sns.countplot(data=df, x="contract_type")
plt.xlabel("Contract type")
plt.ylabel("Customer count")
plt.title("Contract type distribution")
plt.show()
```

**Expected output**

```text
A bar chart comparing Monthly and Annual counts
```

**Explanation**

The plot helps identify category imbalance. It does not show which category has higher churn until the target is included in the analysis.



### 7.5 Inspect missing category values

### 7.5. Include missing values in counts

```python
print(df["city"].value_counts(dropna=False))
```

**Expected output**

```text
Category counts including missing values if present
```

**Explanation**

Including missing values helps reveal incomplete categorical data.



## 8. Univariate analysis — one feature at a time

**Univariate analysis** examines one variable independently.

Examples:
- Age distribution.
- Monthly spend distribution.
- Contract type counts.
- Missing-value count in each column.

### 8.1 Summarize one numeric feature

### 8.1. Summarize monthly spend

```python
print(df["monthly_spend"].describe())
```

**Expected output**

```text
Count, mean, standard deviation, minimum, quartiles and maximum
```

**Explanation**

This summary describes the spend feature only.



### 8.2 Summarize one categorical feature

### 8.2. Summarize contract type

```python
print(df["contract_type"].describe())
```

**Expected output**

```text
Count, unique categories, most frequent category and its frequency
```

**Explanation**

For object/category data, describe returns category-focused summaries rather than numeric statistics.



## 9. Bivariate analysis — relationships between two variables

**Bivariate analysis** studies how two variables behave together.

Examples:
- Monthly spend and churn.
- Contract type and churn.
- Tenure and monthly spend.
- Age and support calls.

### 9.1 Compare average spend by churn label

### 9.1. Group spend by target

```python
spend_by_churn = df.groupby("churn")["monthly_spend"].agg(
    ["count", "mean", "median"]
)

print(spend_by_churn)
```

**Expected output**

```text
A summary of count, mean and median spend for each churn class
```

**Explanation**

This compares groups. It is descriptive evidence, not proof that spending causes churn.



### 9.2 Compare churn rates by contract type

### 9.2. Calculate churn rate by contract

```python
contract_churn = df.groupby("contract_type")["churn"].agg(
    ["count", "mean"]
)

print(contract_churn)
```

**Expected output**

```text
A count and mean churn label for each contract type
```

**Explanation**

When churn is coded 0/1, the mean is the observed churn proportion. State the denominator and sample size when interpreting it.



### 9.3 Visualize spend by churn class

### 9.3. Use a box plot

```python
sns.boxplot(data=df, x="churn", y="monthly_spend")
plt.xlabel("Churn label")
plt.ylabel("Monthly spend")
plt.title("Monthly spend by churn class")
plt.show()
```

**Expected output**

```text
A box plot comparing spend distributions across churn labels
```

**Explanation**

The chart compares medians, spread and potential extreme values. It does not prove a causal relationship.



### 9.4 Visualize tenure by churn class

### 9.4. Compare tenure distributions

```python
sns.violinplot(data=df, x="churn", y="tenure_months", inner="quartile")
plt.xlabel("Churn label")
plt.ylabel("Tenure (months)")
plt.title("Tenure distribution by churn class")
plt.show()
```

**Expected output**

```text
A violin plot comparing tenure distributions
```

**Explanation**

A violin plot estimates distribution shape. With small samples, interpret the smooth shape cautiously.



### 9.5 Compare support calls and churn

### 9.5. Create a grouped summary

```python
support_churn = df.groupby("support_calls")["churn"].agg(
    ["count", "mean"]
)

print(support_churn)
```

**Expected output**

```text
A row for each support-call count with observation count and churn proportion
```

**Explanation**

This can reveal a pattern worth investigating. Check sample size for each group before making conclusions.



## 10. Multivariate analysis — more than two variables

**Multivariate analysis** examines several variables together. A single relationship may change when another variable is considered.

Example business questions:
- Does contract type relate to churn differently across cities?
- How does tenure vary across contract type and churn?
- Do spend and support calls jointly distinguish customer groups?

### 10.1 Group by city and contract type

### 10.1. Summarize multiple categories

```python
multi_summary = df.groupby(
    ["city", "contract_type"]
).agg(
    customer_count=("customer_id", "count"),
    average_spend=("monthly_spend", "mean"),
    churn_rate=("churn", "mean")
)

print(multi_summary)
```

**Expected output**

```text
A grouped summary by city and contract type
```

**Explanation**

The aggregation reports row count, average spend and churn rate for each combination. Small groups can have unstable rates.



### 10.2 Use a pivot table

### 10.2. Create a city-by-contract churn pivot

```python
pivot = df.pivot_table(
    index="city",
    columns="contract_type",
    values="churn",
    aggfunc="mean"
)

print(pivot)
```

**Expected output**

```text
A table of churn rates by city and contract type
```

**Explanation**

The index defines rows, columns defines column groups, values selects the metric and aggfunc specifies the summary.



### 10.3 Compare spend and tenure by churn

### 10.3. Calculate several metrics by target

```python
summary = df.groupby("churn").agg(
    avg_spend=("monthly_spend", "mean"),
    median_spend=("monthly_spend", "median"),
    avg_tenure=("tenure_months", "mean"),
    avg_support_calls=("support_calls", "mean")
)

print(summary)
```

**Expected output**

```text
A summary comparing several metrics across churn groups
```

**Explanation**

This provides a broader view than a single feature comparison. It remains descriptive and should be followed by validation on a larger dataset.



## 11. Feature relationships and scatter plots

A scatter plot places one numerical feature on the horizontal axis and another on the vertical axis. Each point represents an observation.

### 11.1 Plot monthly spend against tenure

### 11.1. Create a scatter plot

```python
sns.scatterplot(
    data=df,
    x="tenure_months",
    y="monthly_spend",
    hue="churn"
)
plt.xlabel("Tenure (months)")
plt.ylabel("Monthly spend")
plt.title("Spend versus tenure by churn")
plt.show()
```

**Expected output**

```text
A scatter plot with points colored by churn label
```

**Explanation**

The hue parameter separates target groups visually. Look for clusters, overlap, unusual points and possible non-linear patterns.



### 11.2 Add a regression trend line for exploration

### 11.2. Use a regression plot

```python
sns.regplot(
    data=df,
    x="tenure_months",
    y="monthly_spend",
    scatter_kws={"alpha": 0.7}
)
plt.xlabel("Tenure (months)")
plt.ylabel("Monthly spend")
plt.title("Tenure and monthly spend")
plt.show()
```

**Expected output**

```text
A scatter plot with a fitted linear trend line
```

**Explanation**

The line summarizes a simple linear trend. It does not establish causation or prove that a linear model is appropriate.



## 12. EDA questions and chart selection

| Analytical question | Suitable first chart |
|---|---|
| What is the distribution of age? | Histogram |
| Are there extreme spend values? | Box plot and histogram |
| How many customers belong to each city? | Count plot |
| How does spend differ by churn? | Box plot or violin plot |
| Is tenure associated with spend? | Scatter plot |
| How does churn rate vary by contract? | Bar chart of group rates with counts |
| How does a numeric feature change over time? | Line plot |
| How do several numerical features relate? | Pair plot or correlation heatmap |

## 13. Missingness and data quality as part of EDA

EDA should inspect data quality before interpreting relationships.

### 13.1 Missing-value counts

### 13.1. Count missing values

```python
print(df.isna().sum())
```

**Expected output**

```text
Missing-value count for every column
```

**Explanation**

This helps identify incomplete features. A missing-value pattern may differ across target groups.



### 13.2 Duplicate rows

### 13.2. Count exact duplicates

```python
print(df.duplicated().sum())
```

**Expected output**

```text
The number of exact duplicate rows
```

**Explanation**

This is an initial check. Repeated customer IDs may be valid if the row grain contains repeated events.



### 13.3 Category cardinality

### 13.3. Count distinct categories

```python
print(df.nunique())
```

**Expected output**

```text
Distinct-value counts for each column
```

**Explanation**

High-cardinality fields may require special treatment. An ID column with many unique values should not automatically be treated as a predictive feature.



## 14. EDA and target leakage

A feature may show a very strong relationship with the target because it was recorded after the target event.

Example: cancellation_reason may strongly distinguish churned customers. But if it is only entered after cancellation, it cannot be used to predict churn before cancellation.

During EDA ask:
- Was the feature recorded before or after the target event?
- Would the feature be available at the moment of prediction?
- Is it a direct restatement of the target?
- Is it generated using future information?

A strong relationship is not automatically a useful feature.

## 15. Turn observations into findings

A useful EDA observation states:
1. What pattern was observed.
2. Which group or feature it concerns.
3. The supporting count or summary.
4. A limitation or follow-up question.

Weak statement:
> Annual customers have lower churn.

Better statement:
> In this sample, the observed churn proportion is lower for annual contracts than monthly contracts.

Responsible interpretation:
> This is a descriptive pattern in a small sample. We need larger, representative data and prediction-time checks before deciding whether contract type is a useful feature.

## 16. Common beginner mistakes

| Mistake | Better approach |
|---|---|
| Create charts without a question | State the question before selecting a plot |
| Interpret only the mean | Inspect median, spread and distribution |
| Ignore group sample sizes | Show counts alongside rates |
| Treat correlation as causation | Describe association and alternative explanations |
| Treat every outlier as an error | Investigate source and business meaning |
| Ignore target definition | Confirm event and prediction horizon |
| Use post-outcome fields | Check prediction-time availability |
| Interpret tiny groups confidently | State uncertainty and sample-size limitations |
| Use inappropriate chart types | Match chart to variable type and question |

## 17. Interview-ready explanations

**What is EDA?**  
EDA is a structured process for understanding data quality, structure, distributions and relationships before modelling. It helps identify problems and formulate useful questions.

**What is the difference between univariate, bivariate and multivariate analysis?**  
Univariate analysis studies one variable, bivariate analysis studies two variables together and multivariate analysis studies several variables together.

**Why use groupby() during EDA?**  
It summarizes metrics within meaningful categories, such as average spend or churn rate by contract type. Include counts so group-level results have context.

**Does a chart prove causation?**  
No. A chart can show patterns or associations. Causal claims require an appropriate study design and evidence.

**Why should EDA consider prediction time?**  
A feature recorded after the target event may create leakage. The model must use only information available when the real prediction is made.

## 18. Scenario questions

1. Annual customers show a lower churn rate, but there are only five annual customers. What should you say?  
   **Answer:** It is a small-sample descriptive pattern. Show the count and avoid a strong conclusion until more representative data is available.

2. A scatter plot shows a relationship between spend and churn. Does spend cause churn?  
   **Answer:** Not from the chart alone. Other variables may explain the association.

3. A column almost perfectly predicts the target. What do you inspect?  
   **Answer:** Its definition and timestamp. It may be a post-outcome field or a direct target proxy.

4. A region has more support calls than another region. Is service quality worse?  
   **Answer:** Not necessarily. Compare rates per customer or per order and account for group size.

5. The histogram of spend has a long right tail. What does that suggest?  
   **Answer:** The distribution is positively skewed. Inspect large values and consider whether robust summaries or a transformation are useful.

## 19. Lesson completion checklist

- [ ] I can define EDA and explain its purpose.
- [ ] I can inspect dataset structure and target distribution.
- [ ] I can describe numeric distributions.
- [ ] I can summarize categorical features.
- [ ] I can perform univariate, bivariate and multivariate analysis.
- [ ] I can choose charts based on questions.
- [ ] I can compare groups with counts and rates.
- [ ] I can distinguish association from causation.
- [ ] I can identify possible target leakage during EDA.
- [ ] I can communicate findings with limitations.

**Next lesson:** Day 15 — Correlation and Multicollinearity.
