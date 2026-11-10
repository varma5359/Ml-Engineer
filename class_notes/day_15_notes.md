# Day 15: Correlation and Multicollinearity

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data Preparation and EDA  
**Level:** Beginner to Intermediate  
**Estimated learning time:** 2–3 hours

## 1. Learning objectives

- Explain correlation as an association between variables.
- Interpret positive, negative and near-zero correlation.
- Distinguish Pearson and Spearman correlation.
- Understand why correlation does not prove causation.
- Create and inspect a correlation matrix.
- Visualize numerical relationships using scatter plots and heatmaps.
- Explain multicollinearity among model input features.
- Use correlation and VIF as diagnostic tools.
- Make feature decisions based on model purpose and validation rather than a fixed threshold.

## 2. What is correlation?

**Correlation** describes the direction and strength of association between two variables.

Examples:
- Advertising spend and sales.
- Delivery time and customer satisfaction.
- Customer tenure and monthly spend.
- Temperature and energy usage.

A positive association means the variables tend to move in the same direction. A negative association means they tend to move in opposite directions. A near-zero linear correlation means the selected correlation measure does not capture a strong linear relationship.

**Important:** Correlation does not prove that one variable causes another.

## 3. Create a practice dataset

### 3.1 Import libraries

### 3.1. Import Pandas, NumPy and plotting libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Ready")
```

**Expected output**

```text
Ready
```

**Explanation**

Pandas and NumPy support data manipulation and numerical calculations. Seaborn and Matplotlib support visualization.



### 3.2 Create sample business data

### 3.2. Create advertising and sales observations

```python
df = pd.DataFrame({
    "advertising_spend": [10, 15, 20, 25, 30, 35, 40, 45],
    "sales": [100, 130, 150, 190, 210, 240, 270, 300],
    "discount": [5, 8, 10, 12, 15, 18, 20, 22],
    "delivery_days": [2, 3, 2, 4, 5, 4, 6, 7],
    "customer_rating": [9, 8, 9, 7, 6, 7, 5, 4]
})

print(df)
```

**Expected output**

```text
An eight-row dataset with five numerical columns
```

**Explanation**

The values are illustrative and intentionally show patterns. Correlation computed on a tiny teaching dataset should not be interpreted as business evidence.



### 3.3 Inspect the data

### 3.3. Check the dataset structure

```python
print(df.shape)
print(df.dtypes)
print(df.describe().T)
```

**Expected output**

```text
Shape, data types and descriptive summaries
```

**Explanation**

Confirm all columns are numeric before calculating a numeric correlation matrix.



## 4. Positive, negative and near-zero relationships

### Positive correlation

As one variable increases, the other tends to increase.

Example: advertising spend and sales may rise together.

### Negative correlation

As one variable increases, the other tends to decrease.

Example: delivery delays may be associated with lower customer ratings.

### Near-zero linear correlation

The variables may show little linear association. A non-linear relationship can still exist.

## 5. Visualize relationships before interpreting a coefficient

### 5.1 Scatter plot for advertising and sales

### 5.1. Plot advertising spend against sales

```python
sns.scatterplot(data=df, x="advertising_spend", y="sales")
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

Each point is one observation. The upward pattern suggests positive association in this example.



### 5.2 Scatter plot for delivery time and rating

### 5.2. Plot delivery days against rating

```python
sns.scatterplot(data=df, x="delivery_days", y="customer_rating")
plt.xlabel("Delivery days")
plt.ylabel("Customer rating")
plt.title("Delivery time versus rating")
plt.show()
```

**Expected output**

```text
A scatter plot with a downward pattern
```

**Explanation**

The points suggest that longer delivery times are associated with lower ratings in this illustrative dataset. Other factors may also influence ratings.



### 5.3 Plot a non-linear pattern

### 5.3. Create a curved relationship

```python
curve_df = pd.DataFrame({
    "x": np.arange(-5, 6),
    "y": np.arange(-5, 6) ** 2
})

sns.scatterplot(data=curve_df, x="x", y="y")
plt.title("A curved relationship")
plt.show()
```

**Expected output**

```text
A U-shaped scatter plot
```

**Explanation**

A clear non-linear relationship can have a near-zero Pearson correlation. Always inspect scatter plots rather than relying on one coefficient.



## 6. Pearson correlation

**Pearson correlation** measures the strength and direction of a linear relationship between two numerical variables.

Its value ranges from -1 to +1:
- +1 indicates perfect positive linear association.
- -1 indicates perfect negative linear association.
- 0 indicates no linear association captured by Pearson's measure.

The coefficient is a summary, not a causal explanation.

### 6.1 Calculate Pearson correlation for two columns

### 6.1. Use corr() for two features

```python
print(df["advertising_spend"].corr(df["sales"]))
```

**Expected output**

```text
A value close to +1 for this illustrative data
```

**Explanation**

The result is high because sales increase consistently as advertising spend increases in the constructed sample.



### 6.2 Calculate a Pearson correlation matrix

### 6.2. Use DataFrame corr()

```python
corr_matrix = df.corr(method="pearson")

print(corr_matrix.round(2))
```

**Expected output**

```text
A square matrix showing pairwise Pearson correlations
```

**Explanation**

Each cell compares two numerical columns. The diagonal is one because each variable is perfectly correlated with itself.



### 6.3 Select correlations with the target

### 6.3. Inspect correlation with one chosen column

```python
sales_corr = corr_matrix["sales"].sort_values(ascending=False)

print(sales_corr)
```

**Expected output**

```text
A sorted list of correlations with sales
```

**Explanation**

This can help prioritize relationships for investigation. High correlation alone does not establish predictive usefulness or causation.



## 7. Spearman rank correlation

**Spearman correlation** measures association using the ranked order of values. It can detect monotonic relationships that are not necessarily linear.

Use it when:
- Values are ordinal.
- A monotonic relationship may be non-linear.
- Outliers make Pearson's linear association less representative.
- The ranking is more meaningful than the exact numeric distance.

### 7.1 Calculate Spearman correlation

### 7.1. Use method='spearman'

```python
spearman_matrix = df.corr(method="spearman")

print(spearman_matrix.round(2))
```

**Expected output**

```text
A rank-correlation matrix
```

**Explanation**

The values compare ranked observations rather than relying on the original numeric distances.



### 7.2 Compare Pearson and Spearman for one pair

### 7.2. Compare two correlation methods

```python
pearson_value = df["advertising_spend"].corr(
    df["sales"],
    method="pearson"
)

spearman_value = df["advertising_spend"].corr(
    df["sales"],
    method="spearman"
)

print(pearson_value)
print(spearman_value)
```

**Expected output**

```text
Two values describing the association using different methods
```

**Explanation**

Differences can occur when relationships are non-linear, ranks and distances behave differently or outliers affect the data.



### 7.3 Compare on a monotonic non-linear example

### 7.3. Create an exponential-like monotonic pattern

```python
nonlinear_df = pd.DataFrame({
    "x": np.arange(1, 11),
    "y": np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
})

print(nonlinear_df.corr(method="pearson").round(2))
print(nonlinear_df.corr(method="spearman").round(2))
```

**Expected output**

```text
Pearson and Spearman matrices; Spearman is 1.00 for the perfectly increasing ranks
```

**Explanation**

The values increase monotonically but not linearly. Spearman captures the perfect rank ordering; Pearson measures linear association in the original values.



## 8. Correlation heatmap

A **heatmap** uses color intensity to display values in a matrix. It can help identify groups of features that are strongly related.

### 8.1 Create a heatmap

### 8.1. Plot the correlation matrix

```python
plt.figure(figsize=(8, 5))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature correlation heatmap")
plt.show()
```

**Expected output**

```text
A heatmap with annotated pairwise correlation values
```

**Explanation**

The annot argument displays values. The color palette maps low and high correlations to different colors. Interpret the numbers and labels, not color alone.



### 8.2 Hide the repeated upper triangle

### 8.2. Create a triangular heatmap

```python
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(8, 5))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f"
)
plt.title("Lower-triangle correlation heatmap")
plt.show()
```

**Expected output**

```text
A heatmap displaying one half of the symmetric correlation matrix
```

**Explanation**

The matrix is symmetric, so the upper and lower triangles repeat the same pairwise values. A mask can reduce visual clutter.



## 9. Correlation is not causation

Suppose advertising spend and sales increase together. It does not prove advertising alone caused sales to increase.

Other possible explanations:
- Seasonal demand increased at the same time.
- Product availability improved.
- A promotion was introduced.
- The business expanded into a new region.
- A general market trend increased both variables.

A correlation can still be useful for prediction, but causal claims require a suitable research design and stronger evidence.

## 10. What is multicollinearity?

**Multicollinearity** occurs when two or more input features contain strong linear relationships with one another.

Example:
- House area in square feet.
- House area in square metres.

These two columns represent almost the same information. In linear models, strong multicollinearity can make individual coefficients unstable or difficult to interpret.

It does not mean every model will fail. Tree-based models and predictive performance may behave differently from coefficient interpretation in linear models.

### 10.1 Create correlated features

### 10.1. Build two related area columns

```python
homes = pd.DataFrame({
    "area_sqft": [800, 1000, 1200, 1400, 1600, 1800],
    "area_sqm": [74.3, 92.9, 111.5, 130.1, 148.6, 167.2],
    "bedrooms": [1, 2, 2, 3, 3, 4],
    "price": [30, 40, 48, 58, 65, 75]
})

print(homes)
```

**Expected output**

```text
A small house dataset with area represented in two units
```

**Explanation**

The two area columns encode nearly identical information in different units.



### 10.2 Calculate correlations among predictors

### 10.2. Inspect the predictor correlation matrix

```python
predictor_corr = homes[["area_sqft", "area_sqm", "bedrooms"]].corr()

print(predictor_corr.round(3))
```

**Expected output**

```text
A matrix showing very high correlation between area_sqft and area_sqm
```

**Explanation**

This suggests redundancy between the two area features. It is a diagnostic signal, not an automatic feature-removal command.



### 10.3 Plot the two area columns

### 10.3. Visualize the relationship

```python
sns.scatterplot(data=homes, x="area_sqft", y="area_sqm")
plt.title("Area in two units")
plt.show()
```

**Expected output**

```text
A near-straight-line scatter plot
```

**Explanation**

The features contain almost the same information because they are unit conversions of one another.



## 11. Variance Inflation Factor (VIF)

**Variance Inflation Factor (VIF)** is a diagnostic used to assess how much a predictor can be explained by other predictors in a linear relationship. High VIF can indicate multicollinearity.

VIF is not a universal automatic feature-selection rule. Interpretation depends on the model, data and purpose.

### 11.1 Prepare predictors

### 11.1. Select numeric predictor columns

```python
X_vif = homes[["area_sqft", "area_sqm", "bedrooms"]].copy()

print(X_vif)
```

**Expected output**

```text
A DataFrame containing three predictors
```

**Explanation**

Do not include the target as a predictor in this diagnostic. Handle missing values and categorical features appropriately before calculating VIF.



### 11.2 Calculate VIF values

### 11.2. Use statsmodels VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_table = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
})

print(vif_table)
```

**Expected output**

```text
A table with VIF values for the selected predictors
```

**Explanation**

The nearly duplicate area columns can produce very high or unstable VIF values. With a tiny teaching dataset, treat results as an illustration only. In practice, use enough observations and inspect numerical stability.



### 11.3 Understand limitations of VIF

- VIF is a linear-predictor diagnostic.
- It does not tell whether a feature is causally important.
- It does not automatically identify the best feature to remove.
- It can be affected by small sample sizes and exact redundancy.
- A high VIF may be acceptable for prediction in some models, while it can complicate coefficient interpretation in linear models.

## 12. What should you do when features are highly correlated?

Possible actions:
1. Check whether two columns are duplicate measurements or unit conversions.
2. Keep the feature with clearer business meaning.
3. Remove redundant features when justified.
4. Combine features if there is a meaningful domain definition.
5. Use regularization for suitable linear models.
6. Compare model stability and validation performance.
7. Keep features if they serve a required business or operational purpose.

Do not remove a feature only because a correlation coefficient crosses an arbitrary threshold.

## 13. Feature correlation with target

A feature can be associated with the target, but this does not prove causality or guarantee useful generalization.

### 13.1 Create a target example

### 13.1. Create a sample feature and target table

```python
churn_df = pd.DataFrame({
    "tenure_months": [2, 4, 6, 8, 12, 18, 24, 30, 36, 48],
    "support_calls": [5, 4, 4, 3, 3, 2, 1, 1, 0, 0],
    "monthly_spend": [1200, 1500, 1800, 2200, 3000, 3500, 4200, 5000, 5500, 6200],
    "churn": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
})

print(churn_df)
```

**Expected output**

```text
A small churn dataset for correlation demonstration
```

**Explanation**

The data is intentionally constructed with clear patterns. It is not suitable for real model conclusions.



### 13.2 Calculate feature-target correlations

### 13.2. Inspect correlations with churn

```python
print(
    churn_df.corr(numeric_only=True)["churn"].sort_values(ascending=False)
)
```

**Expected output**

```text
A sorted list of correlations between numerical columns and churn
```

**Explanation**

This is an exploratory ranking only. Binary target correlation does not replace model evaluation and may be affected by small samples or leakage.



### 13.3 Check feature availability before trusting correlation

A feature may have a high target correlation because it is recorded after the event. Verify its timestamp and definition before considering it for training.

## 14. Common beginner mistakes

| Mistake | Why it is risky | Better approach |
|---|---|---|
| Treat correlation as causation | Other factors may explain the pattern | Describe association only |
| Use Pearson for every relationship | It summarizes linear association | Inspect scatter plots and consider Spearman |
| Assume zero Pearson means no relationship | A curved pattern may exist | Plot the data |
| Remove every highly correlated feature | Some models can use correlated inputs | Consider model purpose and validation |
| Calculate correlation on IDs | Numeric IDs may be labels | Select meaningful numeric features |
| Include target leakage features | High correlation can be misleading | Check prediction-time availability |
| Ignore sample size | Small groups produce unstable patterns | Report counts and uncertainty |
| Use VIF as an automatic delete rule | It is a diagnostic, not a complete decision | Investigate redundancy and coefficient stability |

## 15. Interview-ready explanations

**What is correlation?**  
Correlation describes the direction and strength of association between two variables. It helps identify patterns, but it does not prove causation.

**What is the difference between Pearson and Spearman correlation?**  
Pearson measures linear association using the original values. Spearman measures rank-based monotonic association and can capture relationships that are consistently increasing or decreasing without being linear.

**What is multicollinearity?**  
It occurs when predictor features are strongly linearly related. In linear models, it can make coefficients unstable and difficult to interpret.

**How do you detect multicollinearity?**  
Inspect pairwise predictor correlations, understand feature definitions, visualize relationships and use diagnostics such as VIF when appropriate.

**Should highly correlated features always be removed?**  
No. Consider whether they duplicate information, the model type, interpretability, validation performance and business requirements.

## 16. Scenario questions

1. Advertising spend and sales have high positive correlation. Can you say advertising caused sales to increase?  
   **Answer:** No. Other factors may affect both. Correlation alone does not establish causation.

2. Two house-area columns use square feet and square metres. What is the concern?  
   **Answer:** They contain nearly duplicate information and may create multicollinearity in linear models. Consider retaining the clearer unit.

3. Pearson correlation is near zero but the scatter plot is U-shaped. What does that tell you?  
   **Answer:** There is a non-linear relationship that Pearson does not summarize well.

4. A feature has a very high correlation with churn. What do you check before using it?  
   **Answer:** Verify its meaning and when it is recorded. It may be a post-outcome field or target proxy.

5. VIF is high for two features. Should you remove one immediately?  
   **Answer:** No. Inspect the definitions, sample size, model type and coefficient stability, then compare justified alternatives.

## 17. Lesson completion checklist

- [ ] I can explain positive and negative correlation.
- [ ] I can distinguish Pearson and Spearman.
- [ ] I can create a correlation matrix.
- [ ] I can plot numerical relationships.
- [ ] I understand that correlation does not prove causation.
- [ ] I can explain multicollinearity.
- [ ] I can interpret VIF as a diagnostic.
- [ ] I can investigate redundant features.
- [ ] I can check for leakage before using target correlations.

**Next lesson:** Day 16 — Data Leakage and Preprocessing Pipelines.
