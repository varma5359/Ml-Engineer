# Day 17: Matplotlib and Seaborn for ML Visualization

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain the role of Matplotlib and Seaborn.
- Create and customize common charts.
- Select a chart based on data type and analytical question.
- Visualize numerical distributions and categorical counts.
- Compare feature distributions across target classes.
- Create scatter plots and correlation heatmaps.
- Label axes, units, titles and legends.
- Recognize misleading charts and over-interpretation.
- Use visualization to support EDA and model diagnostics.

## 2. Why do we need visualization in ML?

Tables and summary statistics are important, but charts can reveal patterns that are difficult to see in a table.

Visualization helps us:
- Understand feature distributions.
- Find possible outliers.
- Compare categories.
- See relationships between numerical features.
- Compare features across target groups.
- Communicate findings to technical and business stakeholders.

A chart should answer a question. Do not create charts only because a library provides many plot types.

## 3. Matplotlib and Seaborn

**Matplotlib** is a foundational Python plotting library. It provides detailed control over figures, axes, labels and chart appearance.

**Seaborn** is a statistical visualization library built on Matplotlib. It offers convenient functions for common statistical plots and works naturally with Pandas DataFrames.

### 3.1 Import libraries

### 3.1. Import Matplotlib and Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("Visualization libraries ready")
```

**Expected output**

```text
Visualization libraries ready
```

**Explanation**

The common aliases are `plt` for pyplot and `sns` for Seaborn.



### 3.2 Create a sample dataset

### 3.2. Create business observations

```python
df = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "sales": [120, 150, 135, 180, 210, 205],
    "profit": [20, 25, 18, 35, 42, 39],
    "orders": [30, 35, 33, 42, 48, 46],
    "category": ["A", "B", "A", "B", "A", "B"],
    "churn": [0, 1, 0, 0, 1, 0]
})

print(df)
```

**Expected output**

```text
A six-row sample DataFrame
```

**Explanation**

The data is small and illustrative. The charts teach syntax and interpretation rather than prove business trends.



## 4. Understand the basic Matplotlib workflow

Common plotting steps:
1. Create a figure or axes.
2. Draw the chart.
3. Add a title.
4. Label axes and units.
5. Add a legend when needed.
6. Adjust readability.
7. Call show() in a notebook or interactive script.

### 4.1 Create a simple line plot

### 4.1. Plot monthly sales

```python
plt.plot(df["month"], df["sales"])
plt.title("Monthly sales")
plt.xlabel("Month")
plt.ylabel("Sales (units)")
plt.show()
```

**Expected output**

```text
A line chart showing sales across the six months
```

**Explanation**

A line chart is useful for ordered data such as time. The order of the month values matters.



### 4.2 Add markers

### 4.2. Show individual observations

```python
plt.plot(
    df["month"],
    df["sales"],
    marker="o"
)
plt.title("Monthly sales with markers")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
```

**Expected output**

```text
A line chart with markers at each monthly observation
```

**Explanation**

Markers make the observed points visible. Avoid adding decorative elements that obscure the data.



### 4.3 Add a legend

### 4.3. Label the plotted series

```python
plt.plot(df["month"], df["sales"], marker="o", label="Sales")
plt.legend()
plt.title("Monthly sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
```

**Expected output**

```text
A line chart with a Sales legend
```

**Explanation**

A legend identifies plotted series when more than one series appears.



## 5. Bar charts for category comparisons

A bar chart compares values across categories. It is useful for counts, totals, averages and rates, provided the metric is clearly labelled.

### 5.1 Create a simple bar chart

### 5.1. Plot sales by month

```python
plt.bar(df["month"], df["sales"])
plt.title("Sales by month")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
```

**Expected output**

```text
A vertical bar chart comparing monthly sales
```

**Explanation**

Bars are appropriate for comparing discrete categories or periods. The baseline should be clear when comparing bar heights.



### 5.2 Create a horizontal bar chart

### 5.2. Plot profit horizontally

```python
plt.barh(df["month"], df["profit"])
plt.title("Profit by month")
plt.xlabel("Profit")
plt.ylabel("Month")
plt.show()
```

**Expected output**

```text
A horizontal bar chart of profit by month
```

**Explanation**

Horizontal bars are useful when category labels are long or when vertical labels would overlap.



### 5.3 Sort categories before plotting

### 5.3. Sort categories by sales

```python
sorted_df = df.sort_values("sales", ascending=False)

plt.bar(sorted_df["month"], sorted_df["sales"])
plt.title("Months ranked by sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
```

**Expected output**

```text
A bar chart ordered from highest to lowest sales
```

**Explanation**

Sorting helps compare ranks. For time-series interpretation, keep chronological order instead of ranking months.



## 6. Histograms for numerical distributions

A histogram groups numeric observations into bins and shows how many values fall into each bin.

### 6.1 Create a numeric feature

### 6.1. Create delivery-time observations

```python
delivery = pd.Series([
    18, 20, 22, 25, 26, 27, 29, 30, 31, 35,
    36, 38, 40, 42, 45, 50, 55, 60, 75, 120
])

print(delivery.describe())
```

**Expected output**

```text
A descriptive summary of delivery times
```

**Explanation**

The dataset contains a high value of 120. We will inspect the distribution visually.



### 6.2 Plot a histogram

### 6.2. Create a delivery-time histogram

```python
plt.hist(delivery, bins=8)
plt.title("Delivery-time distribution")
plt.xlabel("Delivery time (minutes)")
plt.ylabel("Number of deliveries")
plt.show()
```

**Expected output**

```text
A histogram showing the distribution of delivery times
```

**Explanation**

The bins group observations. A histogram can reveal skewness and possible unusual values.



### 6.3 Change the number of bins

### 6.3. Use a different bin count

```python
plt.hist(delivery, bins=5)
plt.title("Delivery-time distribution with five bins")
plt.xlabel("Delivery time (minutes)")
plt.ylabel("Count")
plt.show()
```

**Expected output**

```text
A histogram with five bins
```

**Explanation**

The bin count changes the visible level of detail. Too few bins can hide patterns; too many can make the chart noisy.



### 6.4 Use Seaborn histplot

### 6.4. Plot with Seaborn

```python
sns.histplot(data=delivery, bins=8)
plt.title("Delivery-time distribution")
plt.xlabel("Delivery time (minutes)")
plt.ylabel("Count")
plt.show()
```

**Expected output**

```text
A Seaborn histogram
```

**Explanation**

Seaborn provides convenient statistical plotting defaults and integrates with Pandas.



### 6.5 Add a density estimate

### 6.5. Add KDE to the histogram

```python
sns.histplot(data=delivery, bins=8, kde=True)
plt.title("Delivery-time distribution with density")
plt.show()
```

**Expected output**

```text
A histogram with a smooth density estimate
```

**Explanation**

KDE is a smoothed estimate. It depends on bandwidth and sample size and should not be treated as an exact distribution.



## 7. Box plots for spread and unusual values

A box plot summarizes the distribution using quartiles, a median and whiskers. Values outside the whiskers are potential outliers under the plot's convention.

### 7.1 Create a box plot

### 7.1. Plot delivery-time spread

```python
sns.boxplot(x=delivery)
plt.xlabel("Delivery time (minutes)")
plt.title("Delivery-time box plot")
plt.show()
```

**Expected output**

```text
A horizontal box plot with median, quartiles and possible outlier points
```

**Explanation**

The box displays the middle half of the observations. The whiskers follow a statistical convention and do not define business-valid limits.



### 7.2 Compare two groups

### 7.2. Create delivery data by region

```python
region_df = pd.DataFrame({
    "region": ["North"] * 6 + ["South"] * 6,
    "delivery_days": [2, 3, 2, 4, 3, 2, 3, 5, 4, 6, 3, 7]
})

print(region_df)
```

**Expected output**

```text
A dataset containing delivery days for North and South regions
```

**Explanation**

The group labels allow us to compare distributions across regions.



### 7.3. Plot grouped box plots

```python
sns.boxplot(
    data=region_df,
    x="region",
    y="delivery_days"
)
plt.xlabel("Region")
plt.ylabel("Delivery days")
plt.title("Delivery time by region")
plt.show()
```

**Expected output**

```text
Two box plots comparing delivery days by region
```

**Explanation**

Compare medians and spread, and inspect group sizes. Do not assume a difference is statistically or operationally meaningful without further analysis.



## 8. Count plots for categorical features

### 8.1 Create category data

### 8.1. Create payment category values

```python
payments = pd.DataFrame({
    "payment_method": [
        "UPI", "Card", "UPI", "Cash", "Card",
        "UPI", "Cash", "UPI", "Card", "UPI"
    ]
})

print(payments["payment_method"].value_counts())
```

**Expected output**

```text
Frequency counts for UPI, Card and Cash
```

**Explanation**

This gives the number of records in each category.



### 8.2 Create a count plot

### 8.2. Visualize payment method counts

```python
sns.countplot(data=payments, x="payment_method")
plt.title("Orders by payment method")
plt.xlabel("Payment method")
plt.ylabel("Number of orders")
plt.show()
```

**Expected output**

```text
A bar chart showing category counts
```

**Explanation**

countplot counts the observations in each category. It does not require a precomputed count column.



### 8.3 Order categories by frequency

### 8.3. Sort count-plot categories

```python
payment_order = payments["payment_method"].value_counts().index

sns.countplot(
    data=payments,
    x="payment_method",
    order=payment_order
)
plt.title("Payment methods ordered by frequency")
plt.show()
```

**Expected output**

```text
A count plot ordered from most frequent to least frequent
```

**Explanation**

The order parameter makes comparisons easier when there are many categories.



## 9. Scatter plots for relationships

A scatter plot displays paired numerical observations. It helps reveal positive, negative, non-linear and clustered patterns.

### 9.1 Create advertising and sales data

### 9.1. Create paired observations

```python
marketing = pd.DataFrame({
    "advertising_spend": [10, 15, 20, 25, 30, 35, 40, 45],
    "sales": [100, 130, 150, 190, 210, 240, 270, 300]
})

print(marketing)
```

**Expected output**

```text
Eight advertising and sales observations
```

**Explanation**

Each row represents one paired observation. The two values in a row must belong to the same period or entity.



### 9.2 Create a scatter plot

### 9.2. Plot spend versus sales

```python
sns.scatterplot(
    data=marketing,
    x="advertising_spend",
    y="sales"
)
plt.xlabel("Advertising spend")
plt.ylabel("Sales")
plt.title("Advertising spend versus sales")
plt.show()
```

**Expected output**

```text
A scatter plot with an upward pattern
```

**Explanation**

The pattern suggests positive association. It does not prove advertising caused the sales change.



### 9.3 Color points by a category

### 9.3. Use hue to show a segment

```python
marketing["region"] = ["North", "South"] * 4

sns.scatterplot(
    data=marketing,
    x="advertising_spend",
    y="sales",
    hue="region"
)
plt.title("Spend and sales by region")
plt.show()
```

**Expected output**

```text
A scatter plot with different colors for North and South
```

**Explanation**

The hue parameter helps inspect whether the relationship differs across segments. Ensure the category labels and legend are clear.



## 10. Line plots for time trends

A line plot is useful when observations have a meaningful order, commonly dates or time periods.

### 10.1 Create monthly observations

### 10.1. Create a time-series table

```python
monthly = pd.DataFrame({
    "month": pd.date_range("2026-01-01", periods=6, freq="MS"),
    "sales": [100, 120, 115, 140, 155, 150]
})

print(monthly)
```

**Expected output**

```text
Six monthly observations
```

**Explanation**

A date column makes the time order explicit.



### 10.2 Plot monthly sales

### 10.2. Create a time-series line chart

```python
sns.lineplot(data=monthly, x="month", y="sales", marker="o")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly sales trend")
plt.xticks(rotation=30)
plt.show()
```

**Expected output**

```text
A line chart showing monthly sales over time
```

**Explanation**

The chart shows the sequence of observed values. A trend in six points is not enough to establish seasonality.



## 11. Correlation heatmaps

A heatmap represents values in a matrix using color. A correlation heatmap can help identify numerical features that move together.

### 11.1 Calculate a correlation matrix

### 11.1. Create numeric correlation values

```python
numeric_corr = monthly.assign(
    month_number=range(1, 7)
)[["sales", "month_number"]].corr()

print(numeric_corr.round(2))
```

**Expected output**

```text
A two-column correlation matrix
```

**Explanation**

The example demonstrates the calculation. For a larger dataset, select meaningful numerical features and avoid treating identifiers as measured values.



### 11.2 Plot a heatmap

### 11.2. Visualize the correlation matrix

```python
sns.heatmap(
    numeric_corr,
    annot=True,
    fmt=".2f"
)
plt.title("Correlation heatmap")
plt.show()
```

**Expected output**

```text
A heatmap with correlation values
```

**Explanation**

Annotations show the numeric coefficient. Color is only a visual aid; interpret the values and underlying sample.



## 12. Pair plots for multiple feature relationships

A pair plot creates pairwise views of selected numerical features. It can help discover patterns and clusters, but can become crowded with many columns.

### 12.1 Create a small feature sample

### 12.1. Select a few numerical columns

```python
pair_data = pd.DataFrame({
    "age": [22, 25, 31, 29, 42, 35, 28, 50],
    "spend": [1200, 1800, 4500, 3200, 7000, 5200, 2100, 9000],
    "tenure": [2, 5, 18, 12, 36, 24, 6, 48]
})

print(pair_data.head())
```

**Expected output**

```text
A DataFrame with three numerical features
```

**Explanation**

Select a small number of relevant features so the resulting chart remains readable.



### 12.2 Create a pair plot

### 12.2. Use pairplot()

```python
sns.pairplot(pair_data)
plt.show()
```

**Expected output**

```text
A matrix of pairwise scatter plots and diagonal distributions
```

**Explanation**

The diagonal shows each feature's distribution. Off-diagonal plots show pairwise relationships.



## 13. Improve chart readability

A good chart should communicate:
- What is being measured.
- The unit of measurement.
- Which group or time period is shown.
- What each color or symbol means.
- Whether the axis uses a special scale.
- How many observations support the comparison.

### 13.1 Add title and labels

### 13.1. Create a clearly labelled chart

```python
sns.histplot(data=delivery, bins=8)
plt.title("Delivery time distribution")
plt.xlabel("Delivery time (minutes)")
plt.ylabel("Number of deliveries")
plt.show()
```

**Expected output**

```text
A histogram with descriptive title and labelled axes
```

**Explanation**

Use labels that a reader can understand without seeing the code.



### 13.2 Rotate long category labels

### 13.2. Improve category label readability

```python
sns.countplot(data=payments, x="payment_method")
plt.xticks(rotation=30)
plt.title("Orders by payment method")
plt.show()
```

**Expected output**

```text
A count plot with rotated category labels
```

**Explanation**

Rotation can reduce overlap. If there are many categories, consider a horizontal bar chart instead.



### 13.3 Add a legend for target classes

### 13.3. Color points by target class

```python
target_view = pd.DataFrame({
    "tenure": [2, 5, 8, 12, 18, 24, 30, 36],
    "spend": [1200, 1800, 2200, 3000, 3500, 4200, 5000, 5500],
    "churn": [1, 1, 1, 0, 0, 0, 0, 0]
})

sns.scatterplot(data=target_view, x="tenure", y="spend", hue="churn")
plt.xlabel("Tenure (months)")
plt.ylabel("Monthly spend")
plt.title("Customer features by churn class")
plt.show()
```

**Expected output**

```text
A scatter plot with separate target-class colors and legend
```

**Explanation**

The chart can reveal overlap or segment patterns. It does not prove the features cause churn.



## 14. Misleading visualization practices

Avoid:
- Omitting units from axis labels.
- Using a truncated bar-chart axis without explaining it.
- Comparing raw counts when group sizes differ greatly.
- Using a line chart for unordered categories.
- Using too many colors or categories.
- Hiding missing values.
- Claiming causation from a scatter plot.
- Ignoring small sample sizes.
- Changing axes or scales to exaggerate differences.
- Presenting a smoothed density curve as observed data.

## 15. Visualization in ML diagnostics

Visualization is useful after model training too. Later topics will use:
- Confusion matrices.
- ROC and precision-recall curves.
- Residual plots.
- Learning curves.
- Feature importance plots.
- Drift and monitoring charts.

At this stage, focus on visualizing raw features and their relationships before fitting a model.

## 16. Interview-ready explanations

**What is the difference between Matplotlib and Seaborn?**  
Matplotlib is a foundational plotting library with detailed figure and axes control. Seaborn provides higher-level statistical charts and convenient integration with tabular data.

**When do you use a histogram?**  
To inspect the distribution of one numerical variable, including spread, skewness and possible unusual values.

**When do you use a box plot?**  
To compare medians, quartiles, spread and potential extreme observations across one or more groups.

**When do you use a scatter plot?**  
To inspect the relationship between two numerical variables and identify clusters, trends or unusual points.

**Why label axes and units?**  
Without labels and units, a chart may be misinterpreted. A chart should be understandable independently of the code that created it.

## 17. Scenario questions

1. You want to compare the distribution of customer spend. Which chart do you choose?  
   **Answer:** A histogram, optionally supported by a box plot for spread and potential outliers.

2. You want to compare customer counts by contract type. Which chart?  
   **Answer:** A count plot or bar chart.

3. You want to inspect the relationship between tenure and monthly spend. Which chart?  
   **Answer:** A scatter plot.

4. You want to show sales month by month. Which chart?  
   **Answer:** A line chart with months in chronological order.

5. One region has twice the number of returns because it has twice as many orders. What should you compare?  
   **Answer:** Return rate per order, not only the raw return count.

6. A scatter plot shows a clear upward relationship. Can you say X caused Y?  
   **Answer:** No. The plot shows association; other factors or study design may be needed to establish causality.

## 18. Lesson completion checklist

- [ ] I can import Matplotlib and Seaborn.
- [ ] I can create line and bar charts.
- [ ] I can create histograms and box plots.
- [ ] I can create count plots and scatter plots.
- [ ] I can create a correlation heatmap.
- [ ] I can use pairplot for selected numerical features.
- [ ] I can label axes and units.
- [ ] I can select charts based on the analytical question.
- [ ] I can avoid misleading chart interpretations.

**Next lesson:** Day 18 — ML Workflow, Baselines and Reproducibility.
