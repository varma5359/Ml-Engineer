# Day 14: EDA, Distributions and Feature Relationships

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define Exploratory Data Analysis.
- Inspect dataset structure and summary statistics.
- Select charts that answer specific questions.
- Compare categories and numeric distributions.
- Avoid confusing association with causation.

## 1. What Is EDA?
Exploratory Data Analysis (EDA) is the process of inspecting data to understand its structure, quality, distributions, patterns and relationships before modelling.

## 2. First Inspection Syntax
```python
df.head()
df.shape
df.info()
df.describe(include="all")
df.isna().sum()
df.nunique()
```
These commands show example rows, dimensions, types, summary statistics, missing counts and distinct-value counts.

## 3. Three Common EDA Examples
1. Numeric distribution: histogram of order amount.
2. Category comparison: count of orders by payment method.
3. Relationship: scatter plot of advertising spend against sales.

## 4. Grouped Analysis
```python
summary = df.groupby("region")["revenue"].agg(["count", "sum", "mean"])
print(summary)
```
This reports count, total and average revenue per region. Confirm the row grain before interpreting count.

## 5. Choosing a Chart
| Chart | Use |
|---|---|
| Histogram | Numeric distribution |
| Box plot | Spread and potential extreme values |
| Bar/count plot | Category comparison |
| Scatter plot | Relationship between two numeric features |
| Line plot | Change over time |
| Heatmap | Matrix such as correlations |

## 6. Scenario Questions
1. Region A has twice the returns of Region B but also twice the orders. Which measure is fairer? Return rate.
2. A scatter plot shows discount and sales rise together. Does discount cause sales? Not proven; other variables may influence both.
3. A chart has no units or time period. What is missing? Context needed to interpret the result.
