# Day 17: Matplotlib and Seaborn for ML Visualization

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Explain Matplotlib and Seaborn.
- Select charts based on the analytical question.
- Create histograms, box plots, count plots and scatter plots.
- Label charts correctly.
- Recognize misleading visualizations.

## 1. What Are Matplotlib and Seaborn?
Matplotlib is a foundational Python plotting library. Seaborn provides higher-level statistical plots built on Matplotlib.

## 2. Histogram
A histogram groups numeric values into bins and shows how frequently values occur.

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(data=df, x="amount", bins=20)
plt.xlabel("Order amount")
plt.ylabel("Number of orders")
plt.show()
```

## 3. Three Chart Examples
**Example 1 — Histogram:** inspect order amount distribution.
```python
sns.histplot(data=df, x="amount", bins=20)
plt.show()
```

**Example 2 — Box plot:** compare order amounts by region.
```python
sns.boxplot(data=df, x="region", y="amount")
plt.show()
```

**Example 3 — Scatter plot:** inspect relationship between discount and sales.
```python
sns.scatterplot(data=df, x="discount", y="sales")
plt.show()
```

Other useful charts include countplot for category counts, lineplot for time trends and heatmap for correlation matrices.

## 4. Chart Quality
Always label axes, include units and time range, choose readable scales and compare rates when group sizes differ. A chart can show association but does not prove causation.

## 5. Scenario Questions
1. A region has more returns but also more orders. Which plot/measure should you use? Compare return rate, not only count.
2. You want to see the distribution of salary. Which chart? Histogram or box plot.
3. A graph's y-axis starts at 95 and exaggerates a small change. What is the issue? A potentially misleading truncated axis.
