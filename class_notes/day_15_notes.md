# Day 15: Correlation and Multicollinearity

**Pathway:** Machine Learning Engineer  
**Phase:** 2 — Data preparation and EDA  
**Level:** Beginner to Intermediate

## Learning Objectives
- Define correlation and interpret its direction and strength.
- Compare Pearson and Spearman correlation.
- Explain multicollinearity.
- Use correlation matrices and VIF as diagnostics.
- Avoid causal claims from correlation.

## 1. What Is Correlation?
Correlation measures the direction and strength of association between two variables. It ranges from -1 to +1 for common correlation coefficients.

- Positive: variables tend to increase together.
- Negative: one tends to decrease as the other increases.
- Near zero: little association captured by that measure.

## 2. Pearson and Spearman
- Pearson: measures linear association.
- Spearman: measures rank-based monotonic association.

```python
df[["area", "price", "bedrooms"]].corr(method="pearson")
df[["area", "price", "bedrooms"]].corr(method="spearman")
```

**Three examples**
1. Height and weight may show positive association.
2. Temperature and heating demand may show negative association in some contexts.
3. A U-shaped relationship may have near-zero Pearson correlation despite a clear nonlinear pattern.

## 3. What Is Multicollinearity?
Multicollinearity occurs when predictor variables are strongly linearly related to one another. It can make coefficients in linear models unstable and difficult to interpret.

Example: area in square feet and area in square metres represent nearly the same information.

## 4. Diagnostics
- Inspect pairwise correlations among predictors.
- Check domain definitions for duplicate/derived columns.
- Calculate Variance Inflation Factor (VIF) when appropriate.
- Compare coefficient stability.

## 5. Scenario Questions
1. Sales and advertising have correlation 0.9. Does this prove advertising caused sales? No.
2. Two columns store area in different units. What can happen? Multicollinearity and unstable coefficient interpretation.
3. Pearson is near zero but a scatter plot shows a U shape. What might help? Investigate nonlinear relationships and Spearman/domain-based analysis.
